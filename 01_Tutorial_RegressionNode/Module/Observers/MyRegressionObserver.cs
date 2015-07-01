using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.ComponentModel;
using ManagedCuda;

using GoodAI.Core.Utils;
using GoodAI.Core;
using GoodAI.Core.Observers;

namespace RegressionModule.Observers
{
    class MyRegressionObserver : MyNodeObserver<MyRegressionNode>
    {
        [MyBrowsable, Category("Display")]
        public int Size { get; set; }
        private MyCudaKernel m_lineKernel;

        public MyRegressionObserver()
        {
            m_kernel = MyKernelFactory.Instance.Kernel(@"Observers\RegressionObserverKernel");
            m_lineKernel = MyKernelFactory.Instance.Kernel(@"Observers\DrawLineKernel");
            Size = 200;
        }

        protected override void Execute()
        {
            float alpha = Target.Output.Host[0];
            float beta = Target.Output.Host[1];
            float xmin = (float)Math.Floor(Target.XData.Host.Min());
            float xmax = (float)Math.Ceiling(Target.XData.Host.Max());
            float ymin = (float)Math.Floor(Target.YData.Host.Min());
            float ymax = (float)Math.Ceiling(Target.YData.Host.Max());

            float xscale = Size / (xmax - xmin);
            float yscale = Size / (ymax - ymin);
            //float scale = Math.Min(xscale, yscale);

            CudaDeviceVariable<float> VBOvar = new CudaDeviceVariable<float>(VBODevicePointer);
            VBOvar.Memset(0xFFFFFFFF);  //fill white


            m_kernel.SetConstantVariable("D_ALPHA", alpha);
            m_kernel.SetConstantVariable("D_BETA", beta);
            //m_kernel.SetConstantVariable("D_SCALE", scale);
            m_kernel.SetConstantVariable("D_XSCALE", xscale);
            m_kernel.SetConstantVariable("D_YSCALE", yscale);
            m_kernel.SetConstantVariable("D_XMIN", xmin);
            m_kernel.SetConstantVariable("D_YMIN", ymin);
            m_kernel.SetConstantVariable("D_SIZE", Size);

            m_kernel.SetupExecution(Target.ValidFields);
            m_kernel.Run(Target.XData, Target.YData, VBODevicePointer, Target.ValidFields);

            m_lineKernel.SetConstantVariable("D_K", beta);
            m_lineKernel.SetConstantVariable("D_Q", alpha);
            //m_lineKernel.SetConstantVariable("D_SCALE", scale);
            m_lineKernel.SetConstantVariable("D_XSCALE", xscale);
            m_lineKernel.SetConstantVariable("D_YSCALE", yscale);
            m_lineKernel.SetConstantVariable("D_XMIN", xmin);
            m_lineKernel.SetConstantVariable("D_YMIN", ymin);
            m_lineKernel.SetConstantVariable("D_SIZE", Size);

            m_lineKernel.SetupExecution(Size);
            m_lineKernel.Run(VBODevicePointer, Size);
        }

        protected override void Reset()
        {
            base.Reset();
            TextureHeight = TextureWidth = Size;
        }
    }
}
