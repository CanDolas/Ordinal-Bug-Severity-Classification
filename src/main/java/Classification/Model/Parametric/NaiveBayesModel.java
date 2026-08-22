package Classification.Model.Parametric;

import Classification.Attribute.ContinuousAttribute;
import Classification.Attribute.DiscreteAttribute;
import Classification.Instance.Instance;
import Classification.InstanceList.InstanceList;
import Classification.InstanceList.InstanceListOfSameClass;
import Classification.InstanceList.Partition;
import Classification.Parameter.Parameter;
import Math.Vector;
import Math.DiscreteDistribution;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;

public class NaiveBayesModel extends GaussianModel implements Serializable {
    private HashMap<String, Vector> classMeans = null;
    private HashMap<String, Vector> classDeviations = null;
    private HashMap<String, ArrayList<DiscreteDistribution>> classAttributeDistributions = null;

    private boolean isTrained = false; // added

    /**
     * Training algorithm for Naive Bayes algorithm with a continuous data set.
     */
    private void trainContinuousVersion(Partition classLists){
        String classLabel;
        classMeans = new HashMap<>();
        classDeviations = new HashMap<>();
        for (int i = 0; i < classLists.size(); i++){
            classLabel = ((InstanceListOfSameClass) classLists.get(i)).getClassLabel();
            Vector averageVector = classLists.get(i).average().toVector();
            classMeans.put(classLabel, averageVector);
            Vector standardDeviationVector = classLists.get(i).standardDeviation().toVector();
            classDeviations.put(classLabel, standardDeviationVector);
        }
    }

    /**
     * Training algorithm for Naive Bayes algorithm with a discrete data set.
     */
    private void trainDiscreteVersion(Partition classLists){
        classAttributeDistributions = new HashMap<>();
        for (int i = 0; i < classLists.size(); i++){
            classAttributeDistributions.put(((InstanceListOfSameClass) classLists.get(i)).getClassLabel(), classLists.get(i).allAttributesDistribution());
        }
    }

    /**
     * General training method.
     */
    public void train(InstanceList trainSet, Parameter parameters) {
        priorDistribution = trainSet.classDistribution();
        Partition classLists = new Partition(trainSet);
        if (classLists.get(0).get(0).getAttribute(0) instanceof DiscreteAttribute){
            trainDiscreteVersion(classLists);
        } else {
            trainContinuousVersion(classLists);
        }
        isTrained = true; // added
    }

    @Override
    public void loadModel(String fileName) {
        try {
            BufferedReader input = new BufferedReader(new InputStreamReader(Files.newInputStream(Paths.get(fileName)), StandardCharsets.UTF_8));
            int size = loadPriorDistribution(input);
            classMeans = loadVectors(input, size);
            classDeviations = loadVectors(input, size);
            input.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    protected double calculateMetric(Instance instance, String Ci) {
        if (classAttributeDistributions == null) {
            return logLikelihoodContinuous(Ci, instance);
        } else {
            return logLikelihoodDiscrete(Ci, instance);
        }
    }

    private double logLikelihoodContinuous(String classLabel, Instance instance) {
        double xi, mi, si;
        double logLikelihood = Math.log(priorDistribution.getProbability(classLabel));
        for (int i = 0; i < instance.attributeSize(); i++) {
            xi = ((ContinuousAttribute) instance.getAttribute(i)).getValue();
            mi = classMeans.get(classLabel).getValue(i);
            si = classDeviations.get(classLabel).getValue(i);
            if (si != 0){
                logLikelihood += -0.5 * Math.pow((xi - mi) / si, 2);
            }
        }
        return logLikelihood;
    }

    private double logLikelihoodDiscrete(String classLabel, Instance instance) {
        String xi;
        double logLikelihood = Math.log(priorDistribution.getProbability(classLabel));
        ArrayList<DiscreteDistribution> attributeDistributions = classAttributeDistributions.get(classLabel);
        for (int i = 0; i < instance.attributeSize(); i++) {
            xi = ((DiscreteAttribute) instance.getAttribute(i)).getValue();
            logLikelihood += Math.log(attributeDistributions.get(i).getProbabilityLaplaceSmoothing(xi));
        }
        return logLikelihood;
    }

    @Override
    public void saveTxt(String fileName) {
        try {
            PrintWriter output = new PrintWriter(fileName, "UTF-8");
            savePriorDistribution(output);
            saveVectors(output, classMeans);
            saveVectors(output, classDeviations);
            output.close();
        } catch (FileNotFoundException | UnsupportedEncodingException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Computes the class probability distribution for a given instance.
     * This method was added to expose per-class probabilities, which the
     * base Starlang NaiveBayesModel does not provide directly; it is
     * required by the ensemble decomposition strategies (HvL, OvO, OvR).
     */
    @Override
    public HashMap<String, Double> predictProbability(Instance instance) {
        HashMap<String, Double> probabilities = new HashMap<>();
        double total = 0.0;

        HashMap<String, Double> logLikelihoods = new HashMap<>();
        for (String classLabel : priorDistribution.keySet()) {
            double logL = calculateMetric(instance, classLabel);
            logLikelihoods.put(classLabel, logL);
        }

        // log-sum-exp trick for numerical stability
        double maxLog = logLikelihoods.values().stream().mapToDouble(v -> v).max().orElse(0.0);
        for (String classLabel : logLikelihoods.keySet()) {
            double likelihood = Math.exp(logLikelihoods.get(classLabel) - maxLog);
            probabilities.put(classLabel, likelihood);
            total += likelihood;
        }

        // normalize to a valid probability distribution
        for (String classLabel : probabilities.keySet()) {
            probabilities.put(classLabel, probabilities.get(classLabel) / total);
        }

        return probabilities;
    }
}
