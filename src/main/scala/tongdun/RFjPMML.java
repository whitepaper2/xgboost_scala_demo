package tongdun;

import com.google.common.io.CharStreams;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.StructType;
import org.dmg.pmml.PMML;
import org.jpmml.model.JAXBUtil;
import org.jpmml.sparkml.PMMLBuilder;

import javax.xml.transform.stream.StreamResult;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.File;
import java.io.OutputStream;
import java.io.FileOutputStream;

public class RFjPMML {
    private static RFjPMML ourInstance = new RFjPMML();

    public static RFjPMML getInstance() {
        return ourInstance;
    }

    private RFjPMML() {
    }

    public static void main(String[] args) throws Exception {
        System.out.println("Hello World");
//        1、read xml；2、加载model
        SparkSession spark = SparkSession
                .builder()
                .master("local[4]")
                .appName("XGBoost4J-Spark Pipeline Example")
                .getOrCreate();
        InputStream is = new FileInputStream("/Users/pengyuan.li/Documents/code/xgboost_scala_demo/src/main/resources/lr_schema.xml");
        String json = CharStreams.toString(new InputStreamReader(is, "UTF-8"));
        StructType schema = (StructType) DataType.fromJson(json);
        PipelineModel pipelineModel = PipelineModel.load("file:///Users/pengyuan.li/Documents/code/xgboost_scala_demo/src/main/resources/2c_lr");
        PMML pmml = new PMMLBuilder(schema, pipelineModel).build();
// Viewing the result
        try {
            File f = new File("/Users/pengyuan.li/Documents/code/xgboost_scala_demo/src/main/resources/rfjpmml.xml");    // 声明File对象
            // 第2步、通过子类实例化父类对象
            OutputStream out = new FileOutputStream(f);
            JAXBUtil.marshalPMML(pmml, new StreamResult(out));
            out.close();
            JAXBUtil.marshalPMML(pmml, new StreamResult(System.out));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
