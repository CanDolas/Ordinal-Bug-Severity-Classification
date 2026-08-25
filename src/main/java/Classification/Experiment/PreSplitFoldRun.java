package Classification.Experiment;

import Classification.Attribute.AttributeType;
import Classification.DataSet.DataDefinition;
import Classification.DataSet.DataSet;
import Classification.InstanceList.InstanceList;
import Classification.Model.DiscreteFeaturesNotAllowed;
import Classification.Model.Model;
import Classification.Parameter.Parameter;
import Classification.Performance.DetailedClassificationPerformance;
import Classification.Performance.ExperimentPerformance;
import Classification.Performance.Performance;

import java.util.ArrayList;
import java.util.function.Supplier;

/**
 * PreSplitFoldRun
 * ----------------
 * Reads leakage-free fold files pre-split on the Python side and runs the
 * train/test loop for each fold (train -> fit, test -> evaluate).
 *
 * Starlang's built-in StratifiedKFoldCrossValidation is NOT used, since the
 * split, TF-IDF/one-hot fitting, and SMOTE/undersampling were all done
 * per-fold in Python. This class only trains/tests on the ready-made
 * train/test pairs read from disk.
 *
 * File naming convention (matches prepare_folds.py):
 *   train: {prefix}_{condition}_fold{i}_train.data
 *   test : {prefix}_fold{i}_test.data         (shared across conditions)
 *
 * example prefix: "Core_Severity_withcross"
 *
 * NOTE: Each fold needs a FRESH model instance (so training state isn't
 * carried over). The constructor takes a Supplier<Model>; a new
 * model is produced each fold by calling modelFactory.get().
 */
public class PreSplitFoldRun {

    private final int K;
    private final String datasetsDir;
    private final int numFeatures;

    /**
     * @param K           number of folds (e.g. 10)
     * @param datasetsDir directory containing the fold .data files (e.g. "folds_output")
     * @param numFeatures number of features (columns) in each .data file, excluding
     *                    the label. Must match the feature count from prepare_folds.py.
     */
    public PreSplitFoldRun(int K, String datasetsDir, int numFeatures) {
        this.K = K;
        this.datasetsDir = datasetsDir;
        this.numFeatures = numFeatures;
    }

    /**
     * Builds a DataDefinition treating all features as CONTINUOUS.
     * (The TF-IDF + one-hot output is fully numeric.)
     */
    private DataDefinition buildDefinition() {
        ArrayList<AttributeType> attributeTypes = new ArrayList<>();
        for (int i = 0; i < numFeatures; i++) {
            attributeTypes.add(AttributeType.CONTINUOUS);
        }
        return new DataDefinition(attributeTypes);
    }

    private InstanceList loadInstances(String fileName) {
        DataSet ds = new DataSet(buildDefinition(), ",", datasetsDir + "/" + fileName);
        return ds.getInstanceList();
    }

    /**
     * Result holder: both the aggregated ExperimentPerformance (for the mean)
     * and each fold's individual DetailedClassificationPerformance.
     */
    public static class FoldRunResult {
        public final ExperimentPerformance aggregate;
        public final ArrayList<DetailedClassificationPerformance> perFold;
        public FoldRunResult(ExperimentPerformance aggregate,
                             ArrayList<DetailedClassificationPerformance> perFold) {
            this.aggregate = aggregate;
            this.perFold = perFold;
        }
    }

    /**
     * Runs the K-fold loop for a given prefix + condition. Returns both the
     * aggregated result (for the mean) and the per-fold detailed results
     * (for writing to Excel).
     *
     * @param prefix        e.g. "Core_Severity_withcross"
     * @param condition     "processed" | "smote" | "undersampled"
     * @param modelFactory  factory producing a fresh model per fold
     * @param parameter     classifier hyperparameters
     */
    public FoldRunResult executeDetailed(String prefix,
                                         String condition,
                                         Supplier<Model> modelFactory,
                                         Parameter parameter) throws DiscreteFeaturesNotAllowed {
        ExperimentPerformance aggregate = new ExperimentPerformance();
        ArrayList<DetailedClassificationPerformance> perFold = new ArrayList<>();

        for (int i = 0; i < K; i++) {
            String trainFile = prefix + "_" + condition + "_fold" + i + "_train.data";
            String testFile  = prefix + "_fold" + i + "_test.data";

            InstanceList trainSet = loadInstances(trainFile);
            InstanceList testSet  = loadInstances(testFile);

            // fresh model per fold (avoid carrying over training state)
            Model model = modelFactory.get();
            model.train(trainSet, parameter);

            Performance p = model.test(testSet);
            aggregate.add(p);
            if (p instanceof DetailedClassificationPerformance) {
                perFold.add((DetailedClassificationPerformance) p);
            } else {
                perFold.add(null); // unexpected; the runner checks for null
            }
        }

        return new FoldRunResult(aggregate, perFold);
    }

    /**
     * Backward-compatible version: returns only the aggregated ExperimentPerformance.
     */
    public ExperimentPerformance execute(String prefix,
                                         String condition,
                                         Supplier<Model> modelFactory,
                                         Parameter parameter) throws DiscreteFeaturesNotAllowed {
        return executeDetailed(prefix, condition, modelFactory, parameter).aggregate;
    }
}
