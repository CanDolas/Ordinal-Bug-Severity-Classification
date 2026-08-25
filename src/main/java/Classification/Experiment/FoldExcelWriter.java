package Classification.Experiment;

import Classification.Performance.ConfusionMatrix;

import org.apache.poi.ss.usermodel.Cell;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.ss.usermodel.Sheet;
import org.apache.poi.ss.usermodel.Workbook;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.HashMap;

/**
 * FoldExcelWriter
 * ----------------
 * Writes per-fold metrics and confusion matrices to a single Excel file.
 *
 * Produces two sheets:
 *   1) "Metrics"   : one row per (dataset, fold) with Accuracy/WF/MAE/RMSE/LOE/SOE.
 *                    fold = -1 -> MEAN row; Time is only populated on the MEAN row.
 *   2) "ConfMatrix": labeled confusion matrix blocks per (dataset, fold),
 *                    separated by a blank row.
 */
public class FoldExcelWriter {

    private final ArrayList<Object[]> metricRows = new ArrayList<>();
    // confusion matrix blocks: each block is {title, labels, matrix}
    private final ArrayList<Object[]> cmBlocks = new ArrayList<>();

    /** Adds a per-fold metric row. */
    public void addFold(String dataset, int fold, double accuracy, double wf,
                        double mae, double rmse, double loe, double soe) {
        metricRows.add(new Object[]{
                dataset, fold, accuracy * 100, wf * 100, mae, rmse, loe, soe, ""});
    }

    /** Adds the MEAN (average) metric row; Time is only populated here. */
    public void addMean(String dataset, double accuracy, double wf,
                        double mae, double rmse, double loe, double soe, double timeSec) {
        metricRows.add(new Object[]{
                dataset, "MEAN", accuracy * 100, wf * 100, mae, rmse, loe, soe, timeSec});
    }

    /** Adds a confusion matrix block. */
    public void addConfusionMatrix(String dataset, int fold,
                                   ConfusionMatrix cm, ArrayList<String> labels) {
        String title = (fold >= 0) ? (dataset + " | fold " + fold) : dataset;
        // collect the matrix as int[][]
        int n = labels.size();
        int[][] matrix = new int[n][n];
        for (int i = 0; i < n; i++) {
            String actual = labels.get(i);
            for (int j = 0; j < n; j++) {
                String pred = labels.get(j);
                int count = 0;
                if (cm != null && cm.getMatrix().containsKey(actual)) {
                    HashMap<String, Integer> row = cm.getMatrix().get(actual);
                    if (row != null && row.containsKey(pred)) {
                        count = row.get(pred);
                    }
                }
                matrix[i][j] = count;
            }
        }
        cmBlocks.add(new Object[]{title, new ArrayList<>(labels), matrix});
    }

    public void writeToExcel(String fileName) {
        try (Workbook wb = new XSSFWorkbook()) {
            writeMetricsSheet(wb);
            writeConfusionSheet(wb);
            try (FileOutputStream out = new FileOutputStream(fileName)) {
                wb.write(out);
            }
        } catch (Exception e) {
            System.out.println("Excel write error: " + e.getMessage());
        }
    }

    private void writeMetricsSheet(Workbook wb) {
        Sheet sheet = wb.createSheet("Metrics");
        String[] header = {"Dataset", "Fold", "Accuracy(%)", "WeightedF(%)",
                "MAE", "RMSE", "LOE", "SOE", "Time(s)"};
        Row h = sheet.createRow(0);
        for (int c = 0; c < header.length; c++) {
            h.createCell(c).setCellValue(header[c]);
        }
        int r = 1;
        for (Object[] row : metricRows) {
            Row excelRow = sheet.createRow(r++);
            for (int c = 0; c < row.length; c++) {
                Cell cell = excelRow.createCell(c);
                Object v = row[c];
                if (v instanceof Number) {
                    cell.setCellValue(((Number) v).doubleValue());
                } else {
                    cell.setCellValue(String.valueOf(v));
                }
            }
        }
        for (int c = 0; c < header.length; c++) sheet.autoSizeColumn(c);
    }

    @SuppressWarnings("unchecked")
    private void writeConfusionSheet(Workbook wb) {
        Sheet sheet = wb.createSheet("ConfMatrix");
        int r = 0;
        for (Object[] block : cmBlocks) {
            String title = (String) block[0];
            ArrayList<String> labels = (ArrayList<String>) block[1];
            int[][] matrix = (int[][]) block[2];
            int n = labels.size();

            // title row
            sheet.createRow(r++).createCell(0).setCellValue(title);
            // column headers: Actual\Pred + labels
            Row head = sheet.createRow(r++);
            head.createCell(0).setCellValue("Actual\\Pred");
            for (int j = 0; j < n; j++) head.createCell(j + 1).setCellValue(labels.get(j));
            // rows
            for (int i = 0; i < n; i++) {
                Row mr = sheet.createRow(r++);
                mr.createCell(0).setCellValue(labels.get(i));
                for (int j = 0; j < n; j++) {
                    mr.createCell(j + 1).setCellValue(matrix[i][j]);
                }
            }
            r++; // blank row between blocks
        }
    }
}
