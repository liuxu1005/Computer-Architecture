# Computer-Architecture
Tufts comp 140

Course Content:

    Technology Trend: power wall & memory wall => parallelizm
    Instruction Level: pipeline
    Data level: GPU, Vector
    Thread level: Multi processor, memory coherency
    Request Level: Warehouse

Final project: Comparison between CPU and GPU

• Platform  

    CPU: Intel Core i5 at 3.3 GHz, 
    GPU: Nvidia k620 at 1.124Gh with 2 G GPU memory.

• Algorithms  

    Five sequential sorting algo-rithms : shell, merge, quick, insertion and radix sorting,  
    Three parallel algorithms: parallel merge and radix sorting, even-odd and bitonic sorting.
 
• Tools  

    time, Nvidia Visual Profiler. 
    
• Model ideal for GPU is

    Not too small amount of data
    Not too big amount of data 
    Independent data
    Intensive calculation with infrequent memory visit
    It is best to have data fit in shared memory

