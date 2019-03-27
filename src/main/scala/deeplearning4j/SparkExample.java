package deeplearning4j;


import org.apache.camel.component.dataset.DataSet;
import org.apache.hadoop.io.Writable;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.deeplearning4j.spark.datavec.DataVecDataSetFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.datavec.spark.transform.misc.StringToWritablesFunction;


import java.util.List;

public class SparkExample {
    private static Logger log = LoggerFactory.getLogger(MLPMinstTwoLayerExample.class);

    public static void main(String[] args) throws Exception {
        String filePath = "hdfs:///your/path/some_csv_file.csv";
        JavaSparkContext sc = new JavaSparkContext();
        JavaRDD<String> rddString = sc.textFile(filePath);

        RecordReader recordReader = new CSVRecordReader(',');
        JavaRDD rddWritables = rddString.map(new StringToWritablesFunction(recordReader));

        int labelIndex = 5;         //Labels: a single integer representing the class index in column number 5
        int numLabelClasses = 10;   //10 classes for the label
        JavaRDD<DataSet> rddDataSetClassification = rddWritables.map(new DataVecDataSetFunction(labelIndex, numLabelClasses, false));


    }
}
