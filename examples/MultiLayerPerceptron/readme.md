Download and unzip MNIST from
 
http://yann.lecun.com/exdb/mnist/
 
Google protobufs might need to be rebuild, if the target system version does not match the provided version. Rebuild google protobufs with
```
protoc -I=. --cpp_out=. <proto_name>.proto
```
to solve this problem.
