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
    y[y.==0.0] = 10.0
    y = convert(atype, reshape(y, m, 1))
    return (x, y)
end

if !isdefined(:xtrn)
    (xtrn,xtst,ytrn,ytst)=loaddata()
end

h5xtrn, h5ytrn = makeh5(xtrn, ytrn)
h5xtst, h5ytst = makeh5(xtst, ytst)

h5write(joinpath(curdir, "mnist_train.h5"), "data", h5xtrn)
h5write(joinpath(curdir, "mnist_train.h5"), "label", h5ytrn)
h5write(joinpath(curdir, "mnist_test.h5"), "data", h5xtst)
h5write(joinpath(curdir, "mnist_test.h5"), "label", h5ytst)

info("Data processing done.")
