package deeplearning4j.ndarray;


import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

import java.lang.reflect.Array;

public class Nd4jDemo {
    private static Nd4jDemo ourInstance = new Nd4jDemo();

    public static Nd4jDemo getInstance() {
        return ourInstance;
    }

    private Nd4jDemo() {
    }

    public static void main(String[] args) throws Exception {
        /*构造 零矩阵和单位矩阵，符合某种概率分布的数据*/
        INDArray ndOnes = Nd4j.ones(3, 3).addi(3);
        System.out.println(ndOnes);

        INDArray ndZeros = Nd4j.zeros(3, 3);
        System.out.println(ndZeros);

        INDArray ndRandoms = Nd4j.rand(3, 5);
        System.out.println(ndRandoms);

        INDArray ndPdf = Nd4j.randomBernoulli(0.4, 5, 5);
        System.out.println(ndPdf);
        /*从Java格式的数据创建*/
        double[] myDoubleArray = {1, 2, 3, 4, 5, 6};
//        double[] flat = ArrayUtil.flattenDoubleArray(myDoubleArray);
        int[] shape = {3, 2};
        INDArray myArr = Nd4j.create(shape, myDoubleArray);
        System.out.println(myArr);
    }
}
