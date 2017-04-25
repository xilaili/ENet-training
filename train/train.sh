#th run.lua --dataset cfl --datapath data/cfl --model models/encoder.lua --save save/cfl/encoder --imHeight 256 --imWidth 512 --labelHeight 32 --labelWidth 64 --nGPU 1 --cachepath save/cfl/encoder/cache 
th run.lua --dataset cfl --datapath data/cfl --model models/decoder.lua --save save/cfl/decoder --imHeight 256 --imWidth 512 --labelHeight 256 --labelWidth 512 --nGPU 1 --cachepath save/cfl/decoder/cache --CNNEncoder save/cfl/encoder/model-best.net -b 5 

