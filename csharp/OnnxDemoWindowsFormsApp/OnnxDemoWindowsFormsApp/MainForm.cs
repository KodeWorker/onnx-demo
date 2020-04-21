using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Windows.Forms;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Newtonsoft.Json;

namespace OnnxDemoWindowsFormsApp
{
    public partial class MainForm : Form
    {
        private Tensor<float> image = null;
        private InferenceSession session = null;
        List<NamedOnnxValue> inputs = null;
        private List<string> labelMap = null;

        public MainForm()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {

        }

        private void button1_Click(object sender, EventArgs e)
        {
            var fileContent = string.Empty;
            var filePath = string.Empty;

            using (OpenFileDialog openFileDialog = new OpenFileDialog())
            {
                openFileDialog.InitialDirectory = "c:\\";
                openFileDialog.Filter = "image files (*.jpg)|*.jpg|image files (*.png)|*.png|All files (*.*)|*.*";
                openFileDialog.FilterIndex = 0;
                openFileDialog.RestoreDirectory = true;

                if (openFileDialog.ShowDialog() == DialogResult.OK)
                {
                    //Get the path of specified file
                    filePath = openFileDialog.FileName;
                    float[] mean = { 0.485F, 0.456F, 0.406F };
                    float[] stddev = { 0.229F, 0.224F, 0.225F };
                    int[] shape = { 1, 3, 224, 224 };

                    Bitmap bitmapImage = new Bitmap(filePath);
                    Bitmap resizedBitmap;
                    if (shape[1] == 3)
                    {
                        resizedBitmap = Process(bitmapImage, bitmapImage.Width, bitmapImage.Height, shape[3], shape[2]);
                    }
                    else
                    {
                        resizedBitmap = Process(bitmapImage, bitmapImage.Width, bitmapImage.Height, shape[2], shape[1]);
                    }

                    image = ConvertImageToFloatTensor(resizedBitmap, mean, stddev, shape);
                    pictureBox1.Image = resizedBitmap;
                }
            }

            //MessageBox.Show("File Content at path: " + filePath, "info", MessageBoxButtons.OK);
            if(session != null)
            {
                var inputMeta = session.InputMetadata;
                var inputName = inputMeta.First().Key.ToString();
                inputs = new List<NamedOnnxValue>()
                     {
                        NamedOnnxValue.CreateFromTensor<float>(inputName, image),
                     };
            }

        }

        private void button2_Click(object sender, EventArgs e)
        {
            var fileContent = string.Empty;
            var filePath = string.Empty;

            using (OpenFileDialog openFileDialog = new OpenFileDialog())
            {
                openFileDialog.InitialDirectory = "c:\\";
                openFileDialog.Filter = "model files (*.onnx)|*.onnx|All files (*.*)|*.*";
                openFileDialog.FilterIndex = 0;
                openFileDialog.RestoreDirectory = true;

                if (openFileDialog.ShowDialog() == DialogResult.OK)
                {
                    //Get the path of specified file
                    filePath = openFileDialog.FileName;

                    //Read the contents of the file into a stream
                    session = new InferenceSession(filePath);
                    var inputMeta = session.InputMetadata;
                    var inputName = inputMeta.First().Key.ToString();
                    inputs = new List<NamedOnnxValue>()
                     {
                        NamedOnnxValue.CreateFromTensor<float>(inputName, image),
                     };

                }
            }

            //MessageBox.Show("File Content at path: " + filePath, "info", MessageBoxButtons.OK);
            //modelPath = filePath;
        }

        private void button3_Click(object sender, EventArgs e)
        {
            int maxIndex = 0;
            
            using (var results = session.Run(inputs))
            {
                // manipulate the results
                foreach (var r in results)
                {
                    float[] res = r.AsTensor<float>().ToArray();
                    double[] prob = new double[res.Length];
                    for(int i=0; i<prob.Length; i++)
                        prob[i] = Math.Exp(res[i] - res.Max());
                    for (int i = 0; i < prob.Length; i++)
                        prob[i] = prob[i] / prob.Sum();

                    double maxValue = prob.Max();
                    maxIndex = prob.ToList().IndexOf(maxValue);
                }
            }
            string top1 = labelMap[maxIndex];
            MessageBox.Show("Top #1 label: " + top1 + "\n", "info", MessageBoxButtons.OK);
        }

        public static Tensor<float> ConvertImageToFloatTensor(Bitmap bitmapImage, float[] mean, float[] stddev, int[] shape)
        {
            Tensor<float> data = new DenseTensor<float>(shape);
            
            for (int x = 0; x < bitmapImage.Width; x++)
            {
                for (int y = 0; y < bitmapImage.Height; y++)
                {
                    Color color = bitmapImage.GetPixel(x, y);
                    if (shape[1] == 3)
                    {
                        data[0, 0, y, x] = (Convert.ToSingle(color.R) / 255 - mean[0]) / stddev[0];
                        data[0, 1, y, x] = (Convert.ToSingle(color.G) / 255 - mean[1]) / stddev[1];
                        data[0, 2, y, x] = (Convert.ToSingle(color.B) / 255 - mean[2]) / stddev[2];                        
                    }
                    else
                    {
                        data[0, y, x, 0] = (Convert.ToSingle(color.R) / 255 - mean[0]) / stddev[0];
                        data[0, y, x, 1] = (Convert.ToSingle(color.G) / 255 - mean[1]) / stddev[1];
                        data[0, y, x, 2] = (Convert.ToSingle(color.B) / 255 - mean[2]) / stddev[2];
                    }
                }
            }
            return data;
        }

        private static Bitmap Process(Bitmap originImage, int oriwidth, int oriheight, int width, int height)
        {
            Bitmap resizedbitmap = new Bitmap(width, height);
            Graphics g = Graphics.FromImage(resizedbitmap);
            g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.High;
            g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.HighQuality;
            g.Clear(Color.Transparent);
            g.DrawImage(originImage, new Rectangle(0, 0, width, height), new Rectangle(0, 0, oriwidth, oriheight), GraphicsUnit.Pixel);
            return resizedbitmap;
        }

        private void button4_Click(object sender, EventArgs e)
        {
            var fileContent = string.Empty;
            var filePath = string.Empty;

            using (OpenFileDialog openFileDialog = new OpenFileDialog())
            {
                openFileDialog.InitialDirectory = "c:\\";
                openFileDialog.Filter = "label files (*.json)|*.json|All files (*.*)|*.*";
                openFileDialog.FilterIndex = 0;
                openFileDialog.RestoreDirectory = true;

                if (openFileDialog.ShowDialog() == DialogResult.OK)
                {
                    //Get the path of specified file
                    filePath = openFileDialog.FileName;

                    //Read the contents of the file into a stream
                    using (StreamReader r = new StreamReader(filePath))
                    {
                        string json = r.ReadToEnd();
                        labelMap = JsonConvert.DeserializeObject<List<string>>(json);
                    }
                }
            }
        }
    }
}
