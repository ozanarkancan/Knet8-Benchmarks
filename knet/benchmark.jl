# Run the benchmark examples with data preloaded to gpu.
# TODO: cpu comparison
# TODO: minibatches on cpu, transferred to gpu

if false
include(Pkg.dir("Knet/examples/housing.jl"))
info("LinReg preloaded to gpu") # 3.1643
for i=1:6; Housing.main("--fast --seed 1 --epochs 10000 --atype KnetArray{Float32}"); end
info("LinReg on cpu")          # 1.1724
for i=1:6; Housing.main("--fast --seed 1 --epochs 10000 --atype Array{Float32}"); end
info("LinReg copy batches from cpu to gpu")
include("housing2.jl")
for i=1:6; Housing2.main("--fast --seed 1 --epochs 10000"); end
end


if false
include(Pkg.dir("Knet/examples/mnist.jl"))
info("Softmax preloaded to gpu")  # 2.4335
for i=1:6; MNIST.main("--fast --seed 1 --epochs 10"); end
info("Softmax on cpu")            # 3.2953
for i=1:6; MNIST.main("--fast --seed 1 --epochs 10 --atype Array{Float32}"); end
info("Softmax copy batches from cpu to gpu")
include("mnist2.jl")
for i=1:6; MNIST2.main("--fast --seed 1 --epochs 10"); end
end

if false
include(Pkg.dir("Knet/examples/mnist.jl"))
info("MLP preloaded to gpu")  # 3.8960
for i=1:6; MNIST.main("--fast --seed 1 --epochs 10 --hidden 64"); end
info("MLP on cpu")            # 6.4951
for i=1:6; MNIST.main("--fast --seed 1 --epochs 10 --hidden 64 --atype Array{Float32}"); end
info("MLP copy batches from cpu to gpu")
include("mnist2.jl")
for i=1:6; MNIST2.main("--fast --seed 1 --epochs 10 --hidden 64"); end
end

if false
include(Pkg.dir("Knet/examples/lenet.jl"))
info("LeNet preloaded to gpu")  # 3.5941
for i=1:6; LeNet.main("--fast --seed 1 --epochs 1"); end
info("LeNet copy batches from cpu to gpu")
include("lenet2.jl")
for i=1:6; LeNet2.main("--fast --seed 1 --epochs 1"); end
end

if true
include(Pkg.dir("Knet/examples/charlm.jl"))
txt19 = Pkg.dir("Knet/data/19.txt")
!isfile(txt19) && download("http://www.gutenberg.org/files/19/19.txt",txt19)
info("CharLM preloaded to gpu")  # 
include("charlm2.jl")
for i=1:6; CharLM2.main("--fast --seed 1 --winit 0.3 --epochs 1 --data $txt19"); end
# info("CharLM on cpu")  # 38.0374
# for i=1:6; CharLM.main("--fast --seed 1 --winit 0.3 --epochs 1 --data $txt19 --atype Array{Float32}"); end
info("CharLM copy batches from cpu to gpu")  # 2.2508
for i=1:6; CharLM.main("--fast --seed 1 --winit 0.3 --epochs 1 --data $txt19"); end
end
