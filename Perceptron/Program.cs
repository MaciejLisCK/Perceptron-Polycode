using System;
using System.Collections.Generic;
using System.Linq;

namespace Perceptron
{
    class Program
    {
        static Random random;

        static double sigmoid(double x) => 1 / (1 + Math.Exp(-x));

        static double sigmoid_derivative(double x) => x * (1 - x);

        static void Main(string[] args)
        {
            var training_inputs = new int[][]
            {
                new [] { 0,0,1 },
                new [] { 1,1,1 },
                new [] { 1,0,1 },
                new [] { 0,1,1 },
            };

            var training_outputs = new[] { 0, 1, 1, 0 };

            random = new Random(1);

            var synaptic_weights = new double[] { -0.16595599d, 0.44064899d, -0.99977125d }.ToList();

            Console.WriteLine(sigmoid(synaptic_weights[2]));

            Console.WriteLine("Random starting synaptic weights: ");
            Console.WriteLine(String.Join(" ", synaptic_weights));

            List<double> outputs = new List<double>();
            for (int i = 0; i < 50; i++)
            {
                Console.WriteLine($"Iteration {i}");

                outputs = GetOutputs(training_inputs, synaptic_weights);
                var errors = GetErrors(outputs, training_outputs);
                var adjustments = GetAdjustment(errors, outputs);

                synaptic_weights = GetNewSynapticWeights(training_inputs, adjustments, synaptic_weights);

                Console.WriteLine("  outputs:          " + String.Join(" ", outputs));
                Console.WriteLine("  errors:           " + String.Join(" ", errors));
                Console.WriteLine("  adjustments:      " + String.Join(" ", adjustments));
                Console.WriteLine("  synaptic_weights: " + String.Join(" ", synaptic_weights));
            }

            Console.WriteLine("Outputs after training: ");
            Console.WriteLine(String.Join(" ", outputs));

            Console.ReadLine();
        }

        private static List<double> GetAdjustment(List<double> errors, List<double> outputs)
        {
            var adjustments = new List<double>();

            for (int i = 0; i < errors.Count; i++)
            {
                var error = errors[i];
                var output = outputs[i];

                var adjustment = error * sigmoid_derivative(output);

                adjustments.Add(adjustment);
            }

            return adjustments;
        }

        private static List<double> GetErrors(List<double> outputs, int[] training_outputs)
        {
            var errors = new List<double>();
            for (int i = 0; i < outputs.Count(); i++)
            {
                var error = training_outputs[i] - outputs[i];

                errors.Add(error);
            }

            return errors;
        }

        private static List<double> GetOutputs(int[][] training_inputs, IList<double> synaptic_weights)
        {
            var outputs = new List<double>();

            for (int i = 0; i < training_inputs.Length; i++)
            {
                var training_input = training_inputs[i];
                double sum = 0;

                for (int j = 0; j < training_input.Length; j++)
                {
                    var single_input = training_input[j];
                    var weight = synaptic_weights[j];

                    sum += single_input * weight;
                }

                var output = sigmoid(sum);

                outputs.Add(output);
            }

            return outputs;
        }

        private static List<double> GetNewSynapticWeights(int[][] training_inputs, List<double> adjustments, List<double> synaptic_weights)
        {
            var weights = new List<double>(synaptic_weights);

            for (int i = 0; i < training_inputs.Length; i++)
            {
                var training_input = training_inputs[i];
                for (int j = 0; j < training_input.Length; j++)
                {
                    weights[j] += training_input[j] * adjustments[i];
                }
            }

            return weights;
        }

        private static IEnumerable<double> GetWeights()
        {
            var weights = new List<double>();
            for (int i = 0; i < 3; i++)
            {
                var weight = 2 * random.NextDouble() - 1;

                weights.Add(weight);
            }

            return weights;
        }
    }

}
