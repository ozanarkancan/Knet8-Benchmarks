# Run the benchmark examples with data preloaded to gpu.
# TODO: cpu comparison
# TODO: minibatches on cpu, transferred to gpu
load_only=true

length(ARGS)==1 || error("Usage: julia benchmark.jl <name>")

if ARGS[1] == "housing"
info("LinReg preloaded to gpu") # 3.1643
include(Pkg.dir("Knet/examples/housing.jl"))
for i=1:6; gc(); Housing.main("--fast --seed 1 --epochs 10000 --atype KnetArray{Float32}"); end
end

if ARGS[1] == "housingcpu"
info("LinReg on cpu")          # 1.1724
include(Pkg.dir("Knet/examples/housing.jl"))
for i=1:6; gc(); Housing.main("--fast --seed 1 --epochs 10000 --atype Array{Float32}"); end
end

if ARGS[1] == "housingxfer"
info("LinReg copy batches from cpu to gpu")
include("housing2.jl")
for i=1:6; gc(); Housing2.main("--fast --seed 1 --epochs 10000"); end
end

if ARGS[1] == "softmax"
info("Softmax preloaded to gpu")  # 2.4335
include(Pkg.dir("Knet/examples/mnist.jl"))
for i=1:6; gc(); MNIST.main("--fast --seed 1 --epochs 10"); end
end

if ARGS[1] == "softmaxcpu"
info("Softmax on cpu")            # 3.2953
include(Pkg.dir("Knet/examples/mnist.jl"))
for i=1:6; gc(); MNIST.main("--fast --seed 1 --epochs 10 --atype Array{Float32}"); end
end

if ARGS[1] == "softmaxxfer"
info("Softmax copy batches from cpu to gpu")
include("mnist2.jl")
for i=1:6; gc(); MNIST2.main("--fast --seed 1 --epochs 10"); end
end

if ARGS[1] == "mlp"
info("MLP preloaded to gpu")  # 3.8960
include(Pkg.dir("Knet/examples/mnist.jl"))
for i=1:6; gc(); MNIST.main("--fast --seed 1 --epochs 10 --hidden 64"); end
end

if ARGS[1] == "mlpcpu"
info("MLP on cpu")            # 6.4951
include(Pkg.dir("Knet/examples/mnist.jl"))
for i=1:6; gc(); MNIST.main("--fast --seed 1 --epochs 10 --hidden 64 --atype Array{Float32}"); end
end

if ARGS[1] == "mlpxfer"
info("MLP copy batches from cpu to gpu")
include("mnist2.jl")
for i=1:6; gc(); MNIST2.main("--fast --seed 1 --epochs 10 --hidden 64"); end
end

if ARGS[1] == "lenet"
info("LeNet preloaded to gpu")  # 3.5941
include(Pkg.dir("Knet/examples/lenet.jl"))
for i=1:6; gc(); LeNet.main("--fast --seed 1 --epochs 1"); end
end

if ARGS[1] == "lenetxfer"
info("LeNet copy batches from cpu to gpu")
include("lenet2.jl")
for i=1:6; gc(); LeNet2.main("--fast --seed 1 --epochs 1"); end
end

txt19 = Pkg.dir("Knet/data/19.txt")
!isfile(txt19) && download("http://www.gutenberg.org/files/19/19.txt",txt19)

if ARGS[1] == "charlmgpu"
info("CharLM is preloaded to gpu")  # 
include("charlm2.jl")
for i=1:6; gc(); CharLM2.main("--fast --seed 1 --winit 0.3 --epochs 1 --data $txt19"); end
end

if ARGS[1] == "charlm" # default is xfer
info("CharLM copy batches from cpu to gpu")  # 2.2508
include(Pkg.dir("Knet/examples/charlm.jl"))
for i=1:6; gc(); CharLM.main("--fast --seed 1 --winit 0.3 --epochs 1 --data $txt19"); end
end

if ARGS[1] == "charlmcpu"
info("CharLM on cpu")  # 38.0374
include(Pkg.dir("Knet/examples/charlm.jl"))
for i=1:6; gc(); CharLM.main("--fast --seed 1 --winit 0.3 --epochs 1 --data $txt19 --atype Array{Float32}"); end
end
