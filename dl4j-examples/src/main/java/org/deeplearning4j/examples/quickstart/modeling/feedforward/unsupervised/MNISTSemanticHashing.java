/*******************************************************************************
 * Copyright (c) 2020 Marc Strapetz
 * Copyright (c) 2020 Konduit K.K.
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.examples.quickstart.modeling.feedforward.unsupervised;

import java.awt.*;
import java.awt.image.*;
import java.io.*;
import java.util.List;
import java.util.*;
import javax.swing.*;

import org.apache.commons.lang3.tuple.*;
import org.deeplearning4j.datasets.iterator.impl.*;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.*;
import org.deeplearning4j.nn.weights.*;
import org.deeplearning4j.optimize.listeners.*;
import org.jetbrains.annotations.*;
import org.nd4j.linalg.activations.*;
import org.nd4j.linalg.api.ndarray.*;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.*;
import org.nd4j.linalg.dataset.*;
import org.nd4j.linalg.dataset.api.iterator.*;
import org.nd4j.linalg.factory.*;
import org.nd4j.linalg.indexing.conditions.*;
import org.nd4j.linalg.learning.config.*;
import org.nd4j.linalg.lossfunctions.*;

public class MNISTSemanticHashing {

    private static final double EPSILON = 1.0E-6D;

    public static Pair<List<DataSet>, List<DataSet>> createDataSets(int rngSeed) throws IOException {
        final int batchSize = 256;
        final int exampleCount = batchSize * 100;
        DataSetIterator iter = new MnistDataSetIterator(batchSize, exampleCount, false, true, true, rngSeed);

        final Random r = new Random(rngSeed);
        final List<DataSet> dataSetsTrains = new ArrayList<>();
        final List<DataSet> dataSetsTest = new ArrayList<>();
        while (iter.hasNext()) {
            DataSet ds = iter.next();
            SplitTestAndTrain split = ds.splitTestAndTrain(batchSize / 2, r);
            dataSetsTrains.add(split.getTrain());
            dataSetsTest.add(split.getTest());
        }
        return Pair.of(dataSetsTrains, dataSetsTest);
    }

    public static void main(String[] args) throws Exception {
        // Configuration parameters

        final int rngSeed = 12345;
        final int hashSize = 24;
        final IActivation codeLayerActivation = new ActivationBinary();
//        final IActivation codeLayerActivation = Activation.LEAKYRELU.getActivationFunction(); // enable to compare performance with non-binarized activation
        final int codeLayer = 2;
        final boolean sortDigistByHammingDistance = true;

        // Network setup

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(rngSeed)
            .weightInit(WeightInit.XAVIER)
            .updater(new AdaGrad(0.05))
            .activation(Activation.RELU)
            .l2(0.0001)
            .list()
            .layer(new DenseLayer.Builder().nIn(784).nOut(256)
                       .build())
            .layer(new DenseLayer.Builder().nIn(256).nOut(hashSize)
                       .activation(codeLayerActivation)
                       .build())
            .layer(new DenseLayer.Builder().nIn(hashSize).nOut(256)
                       .build())
            .layer(new OutputLayer.Builder().nIn(256).nOut(784)
                       .activation(Activation.LEAKYRELU) // gives nicest results when compared among LEAKYRELU, RELU and SOFTPLUS
                       .lossFunction(LossFunctions.LossFunction.MSE)
                       .build())
            .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.setListeners(Collections.singletonList(new ScoreIterationListener(100000)));

        final Pair<List<DataSet>, List<DataSet>> dataSets = createDataSets(rngSeed);
        final List<DataSet> dataSetsTrains = dataSets.getLeft();
        final List<DataSet> dataSetsTest = dataSets.getRight();
        List<INDArray> featuresTrain = createFeatures(dataSetsTrains);
        List<INDArray> featuresTest = createFeatures(dataSetsTest);
        Map<Integer, List<INDArray>> digitToTrainExamples = createDigitToExamples(dataSetsTrains);
        Map<Integer, List<INDArray>> digitToTestExamples = createDigitToExamples(dataSetsTest);

        int nEpochs = 1000;

        Visualizer trainVisualizer = new Visualizer(1, sortDigistByHammingDistance);
        trainVisualizer.show(false);

        Visualizer testVisualizer = new Visualizer(1, sortDigistByHammingDistance);
        testVisualizer.show(true);

        final ActivationProvider activationProvider = new ActivationProvider() {
            @Override
            public INDArray activate(INDArray input) {
                return net.activate(input, Layer.TrainingMode.TEST);
            }

            @Override
            public INDArray code(INDArray input) {
                return net.feedForwardToLayer(codeLayer, input, false).get(codeLayer);
            }
        };

        final boolean calculateCode = codeLayerActivation instanceof ActivationBinary;
        for (int epoch = 0; epoch < nEpochs; epoch++) {
            for (INDArray data : featuresTrain) {
                net.fit(data, data);
            }

            final double testScore = calculateScore(featuresTest, activationProvider);
            System.out.println("Epoch " + epoch + " completed: " + testScore);

            final Map<Integer, List<Triple<INDArray, INDArray, INDArray>>> digitToTrainReconstruction = calculateReconstructions(digitToTrainExamples, activationProvider, calculateCode);
            trainVisualizer.visualize(digitToTrainReconstruction, "Epoch #" + epoch + " Train: " + calculateScore(featuresTrain, activationProvider));

            final Map<Integer, List<Triple<INDArray, INDArray, INDArray>>> digitToTestReconstruction = calculateReconstructions(digitToTestExamples, activationProvider, calculateCode);
            testVisualizer.visualize(digitToTestReconstruction, "Epoch #" + epoch + " Test: " + testScore);
        }

        Map<Integer, List<Pair<Double, INDArray>>> listsByDigit = new HashMap<>();
        for (int i = 0; i < 10; i++) {
            listsByDigit.put(i, new ArrayList<>());
        }
    }

    public static double calculateScore(List<INDArray> features, ActivationProvider activationProvider) {
        double sumScore = 0;
        double sumCount = 0;
        int exampleCount = 0;
        for (INDArray feature : features) {
            for (int row = 0; row < feature.rows(); row++) {
                if (exampleCount++ > 100) {
                    break;
                }

                final INDArray example = feature.getRows(row);
                final INDArray activate = activationProvider.activate(example);
                assertTrue(example.rows() == 1);
                assertTrue(activate.rows() == 1);
                for (int col = 0; col < example.columns(); col++) {
                    final double expected = example.getDouble(col);
                    final double actual = activate.getDouble(col);
                    sumScore += Math.pow(expected - actual, 2);
                    sumCount += 1;
                }
            }
        }

        return sumScore / sumCount;
    }

    @NotNull
    public static Map<Integer, List<Triple<INDArray, INDArray, INDArray>>> calculateReconstructions(Map<Integer, List<INDArray>> digitToTrainExamples, ActivationProvider activationProvider, boolean calculateCode) {
        final Map<Integer, List<Triple<INDArray, INDArray, INDArray>>> digitToTuples = new HashMap<>();
        for (int digit = 0; digit <= 9; digit++) {
            final List<INDArray> examples = digitToTrainExamples.get(digit);
            for (int index = 0; index < Math.min(Visualizer.ROWS_PER_DIGIT, examples.size()); index++) {
                final INDArray example = examples.get(index);
                final INDArray reconstruction = activationProvider.activate(example);
                final INDArray code = calculateCode ? activationProvider.code(example) : null;
                digitToTuples.computeIfAbsent(digit, integer -> new ArrayList<>()).add(Triple.of(example, reconstruction, code));
            }
        }
        return digitToTuples;
    }

    @NotNull
    public static List<INDArray> createFeatures(List<DataSet> dataSets) {
        List<INDArray> features = new ArrayList<>();
        for (DataSet dataSet : dataSets) {
            final INDArray feature = dataSet.getFeatures();
            features.add(feature);
        }
        return features;
    }

    @NotNull
    public static Map<Integer, List<INDArray>> createDigitToExamples(List<DataSet> dataSets) {
        Map<Integer, List<INDArray>> digitToExamples = new HashMap<>();
        for (DataSet dataSet : dataSets) {
            final INDArray feature = dataSet.getFeatures();
            final INDArray labels = dataSet.getLabels();
            for (int row = 0; row < feature.rows(); row++) {
                INDArray example = feature.getRow(row, true);
                int label = Nd4j.argMax(labels.getRow(row, true), 1).getInt(0);
                digitToExamples.computeIfAbsent(label, integer -> new ArrayList<>()).add(example);
            }
        }
        return digitToExamples;
    }

    private static int getHammingDistance(INDArray code1, INDArray code2) {
        final long length = code1.length();
        assertTrue(length == code2.length());
        int dist = 0;
        for (int i = 0; i < length; i++) {
            dist += Math.abs(code1.getDouble(i) - code2.getDouble(i)) > EPSILON ? 1 : 0;
        }
        return dist;
    }

    private static final void assertTrue(boolean assertion) {
        if (!assertion) {
            throw new RuntimeException();
        }
    }

    public static class Visualizer {
        private static final Dimension SCREEN_SIZE = Toolkit.getDefaultToolkit().getScreenSize();
        private static final int IMAGE_SIZE = 28;
        private static final int ROWS_PER_DIGIT = (SCREEN_SIZE.height - 100) / IMAGE_SIZE;

        private final double imageScale;
        private final boolean sortDigitsBasedOnCodeHammingDistance;
        private final int gridWidth;
        private JFrame frame;
        private JPanel panel;

        public Visualizer(int imageScale, boolean sortDigitsBasedOnCodeHammingDistance) {
            this.imageScale = imageScale;
            this.sortDigitsBasedOnCodeHammingDistance = sortDigitsBasedOnCodeHammingDistance;
            this.gridWidth = 10 * 3;
        }

        public void show(boolean rightPartOfScreen) {
            frame = new JFrame();
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setLocation(rightPartOfScreen ? SCREEN_SIZE.width / 2 : 0, 0);

            panel = new JPanel();
            panel.setLayout(new GridLayout(0, gridWidth));

            frame.add(panel);
            frame.setVisible(true);
        }

        public void visualize(Map<Integer, List<Triple<INDArray, INDArray, INDArray>>> digitToPairs, String title) {
            frame.setTitle(title);

            panel.removeAll();

            final List<Component>[] rows = new List[ROWS_PER_DIGIT];
            for (int digit = 0; digit <= 9; digit++) {
                List<Triple<INDArray, INDArray, INDArray>> tuples = digitToPairs.get(digit);
                if (sortDigitsBasedOnCodeHammingDistance) {
                    tuples = new ArrayList<>(tuples);

                    final INDArray baseCode = tuples.get(0).getRight();
                    if (baseCode != null) {
                        tuples.sort(Comparator.comparingInt(o -> getHammingDistance(baseCode, o.getRight())));
                    }
                }

                if (tuples.size() > ROWS_PER_DIGIT) {
                    tuples = tuples.subList(0, ROWS_PER_DIGIT);
                }
                for (int row = 0; row < ROWS_PER_DIGIT; row++) {
                    List<Component> list = rows[row];
                    if (list == null) {
                        list = new ArrayList<>();
                        rows[row] = list;
                    }
                    if (row >= tuples.size()) {
                        list.add(new JLabel(""));
                        list.add(new JLabel(""));
                        continue;
                    }

                    final Triple<INDArray, INDArray, INDArray> pair = tuples.get(row);
                    list.add(createExampleImage(pair.getLeft()));
                    list.add(createExampleImage(pair.getMiddle()));
                    list.add(createCodeImage(pair.getRight()));
                }
            }

            for (List<Component> row : rows) {
                for (Component component : row) {
                    panel.add(component);
                }
            }

            frame.pack();
        }

        private JLabel createExampleImage(INDArray example) {
            BufferedImage bi = new BufferedImage(IMAGE_SIZE, IMAGE_SIZE, BufferedImage.TYPE_BYTE_GRAY);
            for (int i = 0; i < 784; i++) {
                bi.getRaster().setSample(i % IMAGE_SIZE, i / IMAGE_SIZE, 0, (int)(255 * Math.max(0, example.getDouble(i))));
            }
            ImageIcon orig = new ImageIcon(bi);
            Image imageScaled = orig.getImage().getScaledInstance((int)(imageScale * IMAGE_SIZE), (int)(imageScale * IMAGE_SIZE), Image.SCALE_REPLICATE);
            ImageIcon scaled = new ImageIcon(imageScaled);
            return new JLabel(scaled);
        }

        private JLabel createCodeImage(INDArray code) {
            BufferedImage bi = new BufferedImage(IMAGE_SIZE, IMAGE_SIZE, BufferedImage.TYPE_INT_RGB);
            if (code != null) {
                final long codeLength = code.length();
                final int codeBitsPerRow = (int)Math.ceil(Math.sqrt(codeLength));
                final int imageHeight = IMAGE_SIZE - 2;
                int codeDotSize = imageHeight / codeBitsPerRow;
                int imageWidth = codeDotSize * codeBitsPerRow;
                for (int pos = 0; pos < codeLength; pos++) {
                    final double bit = code.getDouble(pos);
                    int x = (pos * codeDotSize) % imageWidth;
                    int y = codeDotSize * ((pos * codeDotSize) / imageWidth);
                    if (y + codeDotSize > imageHeight) {
                        break;
                    }
                    for (int col = 0; col < codeDotSize; col++) {
                        for (int row = 0; row < codeDotSize; row++) {
                            bi.getRaster().setSample(x + col, y + row, bit < 0 ? 0 : 2, 255);
                        }
                    }
                }
            }

            ImageIcon orig = new ImageIcon(bi);
            Image imageScaled = orig.getImage().getScaledInstance((int)(imageScale * IMAGE_SIZE), (int)(imageScale * IMAGE_SIZE), Image.SCALE_REPLICATE);
            ImageIcon scaled = new ImageIcon(imageScaled);
            return new JLabel(scaled);
        }
    }

    public interface ActivationProvider {
        INDArray activate(INDArray input);

        INDArray code(INDArray input);
    }

    private static class ActivationBinary extends BaseActivationFunction {
        public INDArray getActivation(INDArray in, boolean training) {
            in.replaceWhere(Nd4j.ones(in.length()).muli(-1), new LessThan(0));
            in.replaceWhere(Nd4j.ones(in.length()), new GreaterThanOrEqual(0));
            return in;
        }

        public org.nd4j.common.primitives.Pair<INDArray, INDArray> backprop(INDArray in, INDArray epsilon) {
            this.assertShape(in, epsilon);
            Nd4j.getExecutioner().execAndReturn(new TanhDerivative(in, epsilon, in)); // tanh's gradient is a reasonable approximation
            //noinspection rawtypes
            return new org.nd4j.common.primitives.Pair(in, (Object)null);
        }

        @Override
        public int hashCode() {
            return 1;
        }

        @Override
        public boolean equals(Object obj) {
            return obj instanceof ActivationBinary;
        }

        @Override
        public String toString() {
            return "Binary";
        }
    }
}
