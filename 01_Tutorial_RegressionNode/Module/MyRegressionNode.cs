using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YAXLib;

using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Observers;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;

namespace RegressionModule
{
    /// <author>Michal Vlasák</author>
    /// <status>Work in progress</status>
    /// <summary>Regression node</summary>
    /// <description>Will perform online linear regression</description>
    class MyRegressionNode : MyWorkingNode
    {
        [MyInputBlock(0)]
        public MyMemoryBlock<float> XInput
        {
            get { return GetInput(0); }
        }

        [MyInputBlock(1)]
        public MyMemoryBlock<float> YInput
        {
            get { return GetInput(1); }
        }

        [MyOutputBlock(0)]
        public MyMemoryBlock<float> Output
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        [MyPersistable]
        public MyMemoryBlock<float> XData { get; private set; }
        [MyPersistable]
        public MyMemoryBlock<float> YData { get; private set; }

        [MyBrowsable, Category("Params"), YAXSerializableField(DefaultValue = 100)]
        public int BufferSize { get; set; }
        [MyBrowsable, Category("Params"), YAXSerializableField(DefaultValue = 0)]
        public int ValidFields { get; set; }

        public MyGatherDataTask GetData { get; private set; }
        public MyComputeTask Compute { get; private set; }

        public override void UpdateMemoryBlocks()
        {
            XData.Count = BufferSize;
            YData.Count = BufferSize;
            Output.Count = 2;
        }

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);
            validator.AssertError(XInput.Count != 0 && YInput.Count != 0, this, "Both inputs should have size greater than 1.");
            validator.AssertWarning(XInput.Count == 1 && YInput.Count == 1, this, "Inputs should have size 1. Only the first value will be considered.");
        }
    }

    [Description("Gather data")]
    class MyGatherDataTask : MyTask<MyRegressionNode>
    {

        private int m_cursor;

        public override void Init(int nGPU)
        {
            m_cursor = Owner.ValidFields = 0;
        }

        public override void Execute()
        {
            Owner.XInput.CopyToMemoryBlock(Owner.XData, 0, m_cursor, 1);
            Owner.YInput.CopyToMemoryBlock(Owner.YData, 0, m_cursor, 1);
            m_cursor = (m_cursor + 1) % Owner.BufferSize;

            if (Owner.ValidFields != Owner.BufferSize)
            {
                Owner.ValidFields = m_cursor + 1;
            }
        }
    }

    [Description("Compute model")]
    class MyComputeTask : MyTask<MyRegressionNode>
    {
        public override void Init(int nGPU)
        {

        }

        public override void Execute()
        {
            Owner.XData.SafeCopyToHost();
            Owner.YData.SafeCopyToHost();
            float[] X = Owner.XData.Host;
            float[] Y = Owner.YData.Host;
            int N = Owner.ValidFields;

            double x, y;
            double sumX = 0;
            double sumXX = 0;
            double sumY = 0;
            double sumYY = 0;
            double sumProds = 0;


            for (int i = 0; i < N; ++i)
            {
                x = X[i];
                y = Y[i];
                sumProds += x * y;
                sumX += x;
                sumXX += x * x;
                sumY += y;
                sumYY += y * y;
            }

            double meanX = sumX / N;
            double meanY = sumY / N;
            double sx = Math.Sqrt((sumXX - N * meanX * meanX) / (N - 1));
            double sy = Math.Sqrt((sumYY - N * meanY * meanY) / (N - 1));
            double r = (sumProds - N * meanX * meanY) / ((N - 1) * sx * sy);

            double beta = r * (sy / sx);
            double alpha = meanY - beta * meanX;

            Owner.Output.Host[0] = (float)alpha;
            Owner.Output.Host[1] = (float)beta;
            //MyLog.INFO.WriteLine(r);
            Owner.Output.SafeCopyToDevice();
        }
    }
}

