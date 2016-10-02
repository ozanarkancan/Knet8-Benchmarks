if !isdefined(:MNIST)
    local lo=isdefined(:load_only)
    load_only=true
    include(Pkg.dir("Knet/examples/mnist.jl"))
    load_only=lo
end

"""

This example learns to classify hand-written digits from the
[MNIST](http://yann.lecun.com/exdb/mnist) dataset.  There are 60000
training and 10000 test examples. Each input x consists of 784 pixels
representing a 28x28 image.  The pixel values are normalized to
[0,1]. Each output y is converted to a ten-dimensional one-hot vector
(a vector that has a single non-zero component) indicating the correct
class (0-9) for a given image.  10 is used to represent 0.

You can run the demo using `julia lenet.jl`.  Use `julia lenet.jl
--help` for a list of options.  The dataset will be automatically
downloaded.  By default the [LeNet](http://yann.lecun.com/exdb/lenet)
convolutional neural network model will be trained for 10 epochs.  The
accuracy for the training and test sets will be printed at every epoch
and optimized parameters will be returned.

"""
module LeNet2
using Knet,ArgParse
using Main.MNIST: minibatch, xtrn, ytrn, xtst, ytst


function main(args=ARGS)
    s = ArgParseSettings()
    s.description="lenet.jl (c) Deniz Yuret, 2016. The LeNet model on the MNIST handwritten digit recognition problem from http://yann.lecun.com/exdb/mnist."
    s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
        ("--seed"; arg_type=Int; default=-1; help="random number seed: use a nonnegative int for repeatable results")
        ("--batchsize"; arg_type=Int; default=100; help="minibatch size")
        ("--lr"; arg_type=Float64; default=0.1; help="learning rate")
        ("--fast"; action=:store_true; help="skip loss printing for faster run")
        ("--epochs"; arg_type=Int; default=3; help="number of epochs for training")
        ("--gcheck"; arg_type=Int; default=0; help="check N random gradients per parameter")
    end
    println(s.description)
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    println("opts=",[(k,v) for (k,v) in o]...)
    o[:seed] > 0 && srand(o[:seed])
    gpu() >= 0 || error("LeNet only works on GPU machines.")

    dtrn = minibatch4(xtrn, ytrn, o[:batchsize]; atype=Array{Float32})
    dtst = minibatch4(xtst, ytst, o[:batchsize]; atype=Array{Float32})
    w = weights()
    println((:epoch,0,:trn,accuracy(w,dtrn),:tst,accuracy(w,dtst)))

    if o[:fast]
        @time train(w, dtrn; lr=o[:lr], epochs=o[:epochs])
        println((:epoch,o[:epochs],:trn,accuracy(w,dtrn),:tst,accuracy(w,dtst)))
    else
        @time for epoch=1:o[:epochs]
            train(w, dtrn; lr=o[:lr], epochs=1)
            println((:epoch,epoch,:trn,accuracy(w,dtrn),:tst,accuracy(w,dtst)))
            if o[:gcheck] > 0
                gradcheck(loss, w, first(dtrn)...; gcheck=o[:gcheck])
            end
        end
    end
    return w
end

function train(w, data; lr=.1, epochs=20, nxy=0)
    for epoch=1:epochs
        for (x,y) in data
            xx = KnetArray(x); yy = KnetArray(y)
            g = lossgradient(w, xx, yy)
            for i in 1:length(w)
                w[i] -= lr * g[i]
            end
        end
    end
    return w
end

function predict(w,x0)                       # 28,28,1,100
    x1 = pool(relu(conv4(w[1],x0) .+ w[2])) # 12,12,20,100
    x2 = pool(relu(conv4(w[3],x1) .+ w[4])) # 4,4,50,100
    x3 = relu(w[5]*mat(x2) .+ w[6])              # 500,100
    x4 = w[7]*x3 .+ w[8]                     # 10,100
end

function loss(w,x,ygold)
    ypred = predict(w,x)
    ynorm = logp(ypred,1)  # ypred .- log(sum(exp(ypred),1))
    -sum(ygold .* ynorm) / size(ygold,2)
end

lossgradient = grad(loss)

function weights(;ftype=Float32,atype=KnetArray)
    w = Array(Any,8)
    w[1] = xavier(Float32,5,5,1,20)
    w[2] = zeros(Float32,1,1,20,1)
    w[3] = xavier(Float32,5,5,20,50)
    w[4] = zeros(Float32,1,1,50,1)
    w[5] = xavier(Float32,500,800)
    w[6] = zeros(Float32,500,1)
    w[7] = xavier(Float32,10,500)
    w[8] = zeros(Float32,10,1)
    return map(a->convert(atype,a), w)
end

function minibatch4(x, y, batchsize; atype=KnetArray{Float32})
    data = minibatch(x,y,batchsize; atype=atype)
    for i=1:length(data)
        (x,y) = data[i]
        data[i] = (reshape(x, (28,28,1,batchsize)), y)
    end
    return data
end

function accuracy(w, dtst; nxy=0)
    ncorrect = ninstance = nloss = 0
    for (x, ygold) in dtst
        xx = KnetArray(x); yygold = KnetArray(ygold)
        ypred = predict(w, xx)
        ynorm = ypred .- log(sum(exp(ypred),1))
        nloss += -sum(yygold .* ynorm)
        ncorrect += sum((ypred .== maximum(ypred,1)) .* (yygold .== maximum(yygold,1)))
        ninstance += size(yygold,2)
    end
    return (ncorrect/ninstance, nloss/ninstance)
end

function xavier(a...)
    w = rand(a...)
     # The old implementation was not right for fully connected layers:
     # (fanin = length(y) / (size(y)[end]); scale = sqrt(3 / fanin); axpb!(rand!(y); a=2*scale, b=-scale)) :
    if ndims(w) < 2
        error("ndims=$(ndims(w)) in xavier")
    elseif ndims(w) == 2
        fanout = size(w,1)
        fanin = size(w,2)
    else
        fanout = size(w, ndims(w)) # Caffe disagrees: http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1XavierFiller.html#details
        fanin = div(length(w), fanout)
    end
    # See: http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
    s = sqrt(2 / (fanin + fanout))
    w = 2s*w-s
end


# This allows both non-interactive (shell command) and interactive calls like:
# $ julia lenet.jl --epochs 10
# julia> LeNet.main("--epochs 10")
!isinteractive() && (!isdefined(Main,:load_only) || !Main.load_only) && main(ARGS)

end # module

# SAMPLE RUN 65f57ff+ Wed Sep 14 10:02:30 EEST 2016
#
# lenet.jl (c) Deniz Yuret, 2016. The LeNet model on the MNIST handwritten digit recognition problem from http://yann.lecun.com/exdb/mnist.
# opts=(:seed,-1)(:batchsize,100)(:epochs,3)(:lr,0.1)(:gcheck,0)(:fast,true)
# ..................  
# 9.319163 seconds (5.84 M allocations: 277.927 MB, 7.37% gc time)
