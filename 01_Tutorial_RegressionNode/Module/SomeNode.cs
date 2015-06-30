using System;
using System.ComponentModel;
using System.Linq;

using BrainSimulator;
using BrainSimulator.Memory;
using BrainSimulator.Nodes;
using BrainSimulator.Task;
using BrainSimulator.Utils;
using ManagedCuda;
using YAXLib;

namespace NewModule
{
    /// <author>An Engineer</author>
    /// <status>Working sample</status>
    /// <summary>Almost minimal node that just increments each item by a constant.</summary>
    /// <description>
    /// Just a sample node trying to demonstrate how to get started.
    /// </description>
    public class SomeNode : MyWorkingNode
    {
        /// <summary>
        /// Base value used by the IncrementTask.
        /// </summary>
        [MyBrowsable, Category("SomeCategory"), YAXSerializableField(DefaultValue = 0.0f)]
        public float IncrementBase { get; set; }

        [MyInputBlock(0)]
        public MyMemoryBlock<float> Input
        {
            get { return GetInput(0); }
        }

        [MyOutputBlock(0)]
        public MyMemoryBlock<float> Output
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        public int InputCount
        {
            get { return (Input != null) ? Input.Count : 0; }
        }

        public override void UpdateMemoryBlocks()
        {
            Output.Count = InputCount;
        }

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);
            validator.AssertError(InputCount != 0, this, "Zero input size is not allowed.");
        }

        public MyIncrementTask IncrementTask { get; private set; }
    }
    
    /// <summary>
    /// Task that increments all data items by a constant calculated as node's IncrementBase + task's Increment
    /// </summary>
    [Description("Increment all by a constant")]
    public class MyIncrementTask : MyTask<SomeNode>
    {
        /// <summary>
        /// Value that is added to Node's IncrementBase to get the total increment
        /// </summary>
        [MyBrowsable, Category("SomeCategory"), YAXSerializableField(DefaultValue = 1.0f)]
        public float Increment { get; set; }

        private MyCudaKernel m_kernel;

        public override void Init(int nGPU)
        {
            m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"SomeNode", "IncrementAll");
        }

        public override void Execute()
        {
            m_kernel.SetupExecution(Owner.InputCount);
            m_kernel.Run(Owner.Input, Owner.Output, Owner.IncrementBase + Increment, Owner.InputCount);
        }
    }
}
