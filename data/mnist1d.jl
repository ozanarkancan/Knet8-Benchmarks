using HDF5, GZip, Compat

curdir = pwd()

function loaddata()
    info("Loading MNIST...")
    gzload("train-images-idx3-ubyte.gz")[17:end],
    gzload("t10k-images-idx3-ubyte.gz")[17:end],
    gzload("train-labels-idx1-ubyte.gz")[9:end],
    gzload("t10k-labels-idx1-ubyte.gz")[9:end]
end

function gzload(file; path=joinpath(curdir, file), url="http://yann.lecun.com/exdb/mnist/$file")
    isfile(path) || download(url, path)
    f = gzopen(path)
    a = @compat read(f)
    close(f)
    return(a)
end

function makeh5(x, y; n=784, xscale=255, atype=Array{Float32})
    m = length(y)
    x = convert(atype, reshape(transpose(reshape(x, n, m)), m, n, 1, 1)) / xscale
    x = permutedims(x, [2, 3, 4, 1])
    y = convert(Array{Int32}, reshape(y, m, 1, 1, 1))
    y = permutedims(y, [4, 3, 2, 1])
    return (x, y)
end

if !isdefined(:xtrn)
    (xtrn,xtst,ytrn,ytst)=loaddata()
end

xtrn, ytrn = makeh5(xtrn, ytrn)
xtst, ytst = makeh5(xtst, ytst)
println(size(xtrn))
# Caffe has HDF5 input restrictions, so we need to split dataset to smaller parts
i, partlen = 1, 10000
while i <= 60000 / partlen
    lower, upper = (i-1)*partlen+1, i*partlen
    h5write(joinpath(curdir, "mnist_train$(i).h5"), "data", xtrn[:,:,:,lower:upper])
    h5write(joinpath(curdir, "mnist_train$(i).h5"), "label", ytrn[:,:,:,lower:upper])
    i += 1
end

h5write(joinpath(curdir, "mnist_test.h5"), "data", xtst)
h5write(joinpath(curdir, "mnist_test.h5"), "label", ytst)

info("Data processing done.")
