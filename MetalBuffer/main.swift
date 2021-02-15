//
//  main.swift
//  MetalBuffer
//
//  Created by Shogo Nobuhara on 2021/02/16.
//

import Metal

class Compute {
    var device: MTLDevice? = nil
    var cmdQueue: MTLCommandQueue? = nil
    
    init() {
        if let device = MTLCreateSystemDefaultDevice() {
            self.device = device
            self.cmdQueue = device.makeCommandQueue()
        }
    }
    
    
    // GPUでの処理を行う
    func execute(srcBuf: MTLBuffer, dstBuf: MTLBuffer, count: Int)
    {
        let cmdBuf = self.cmdQueue?.makeCommandBuffer()
        let encoder = cmdBuf?.makeComputeCommandEncoder()
        
        // パイプラインを作成する
        guard let pipeline = makeReversePipeline() else {
            return
        }
        
        encoder?.setComputePipelineState(pipeline)
        
        // コンピュートパスにバッファをセットする
        encoder?.setBuffer(srcBuf, offset: 0, index: kReverseIndexSrc)
        encoder?.setBuffer(dstBuf, offset: 0, index: kReverseIndexDst)
        
        var n = Int32(count)
        let countBuf = self.device?.makeBuffer(bytes: &n,
                                               length: MemoryLayout<Int32>.size,
                                               options: .storageModeShared)
        
        
        // 実行コマンドをエンコードする
        self.dispatch(encoder: encoder, pipeline: pipeline, count: count)
        
        
        encoder?.endEncoding()
        cmdBuf?.commit()
        
        
        // 完了待ち
        cmdBuf?.waitUntilCompleted()
        
    }
    
    // パイプライン状態オブジェクトを作成する
    func makeReversePipeline() -> MTLComputePipelineState? {
        // カーネル関数を取得する
        let lib  = self.device?.makeDefaultLibrary()
        guard let kernelFunc = lib?.makeFunction(name: "reverseValuses") else {
            return nil
        }
        
        do {
            return try self.device?.makeComputePipelineState(function: kernelFunc)
        } catch _ {
            return nil
        }
    }
    
    // 実行コマンドをエンコードする
    func dispatch(encoder: MTLComputeCommandEncoder?,
                  pipeline: MTLComputePipelineState,
                  count: Int) {
        guard let device = self.device else {
            return
        }
        
        let w = pipeline.threadExecutionWidth
        let perGroup = MTLSize(width: w, height: 1, depth: 1)
        
        if device.supportsFamily(.common3) {
            // non-uniformスレッドグループに対応している
            let perGrid = MTLSize(width: count, height: 1, depth: 1)
            encoder?.dispatchThreads(perGrid, threadsPerThreadgroup: perGroup)
        } else {
            // non-uniformスレッドグループに非対応
            let groupsPerGrid = MTLSize(width: (count + w - 1), height: 1, depth: 1)
            encoder?.dispatchThreadgroups(groupsPerGrid, threadsPerThreadgroup: perGroup)
        }
        
    }
    
    // 入力バッファを作る
    func makeSrcBuffer(values: [Int32]) -> MTLBuffer? {
        // プレイベートストレーぞモードのバッファを作成する
        let bufSize = MemoryLayout<Int32>.size * values.count
        let resultBuf = self.device?.makeBuffer(length: bufSize, options: .storageModePrivate)
        
        // valuesの内容をコピーしたバッファを作る
        let options: MTLResourceOptions = [.storageModeShared,
                                           .cpuCacheModeWriteCombined]
        
        let tempBuf = self.device?.makeBuffer(bytes: values,
                                              length: bufSize,
                                              options: options)
        
        // メモリコピーコマンドをエンコードする
        let cmdBuf = self.cmdQueue?.makeCommandBuffer()
        let encoder = cmdBuf?.makeBlitCommandEncoder()
        
        if resultBuf != nil && tempBuf != nil {
            encoder?.copy(from: tempBuf!, sourceOffset: 0,
                           to: resultBuf!, destinationOffset: 0,
                           size: bufSize)
        }
        
        encoder?.endEncoding()
        cmdBuf?.commit()
        
        
        return resultBuf
    }
    
    // 出力バッファを作る
    func makeDstBuffer(count: Int) -> MTLBuffer? {
        // プライベートストレージモードのバッファを作成する
        let bufSize = MemoryLayout<Int32>.size * count
        return self.device?.makeBuffer(length: bufSize, options: .storageModePrivate)
    }
    
    // バッファからInt32の配列を読み込む
    func makeInt32Array(buffer: MTLBuffer) -> [Int32] {
        // CPUから読めるテンポラリバッファを作る
        guard let tempBuf = self.device?.makeBuffer(length: buffer.length,
                                                    options: .storageModeShared) else {
            return [Int32]()
        }
        
        // 出力バッファからテンポラリバッファへコピーする
        let cmdBuf = self.cmdQueue?.makeCommandBuffer()
        let encoder = cmdBuf?.makeBlitCommandEncoder()
        
        encoder?.copy(from:buffer,sourceOffset: 0,
                      to: tempBuf,destinationOffset: 0,
                      size: buffer.length)
        
        // コマンド実行
        encoder?.endEncoding()
        cmdBuf?.commit()
        
        // 完了待ち
        cmdBuf?.waitUntilCompleted()
        
        
        // テンポラリバッファの内容を読み込む
        let count = buffer.length / MemoryLayout<Int32>.size
        var result = [Int32](repeating:0,count: count)
        let bufPtr = tempBuf.contents().bindMemory(to: Int32.self, capacity: count)
        
        for i in 0 ..< count {
            result[i] = bufPtr[i]
        }
        
        return result
    }
}


struct Sample {
    static func main() {
        let compute = Compute()
        
        // 入力もとの数列作成
        var srcArray = [Int32](repeating: 0, count: 100)
        for i in 0 ..< srcArray.count {
            srcArray[i] = Int32(i)
        }
        
        // 入力バッファを作成する
        guard let srcBuf = compute.makeSrcBuffer(values: srcArray) else {
            return
        }
        
        // 出力バッファを作成する
        guard let dstBuf = compute.makeDstBuffer(count: srcArray.count) else {
            return
        }
        
        // コンピュートパスを実行する
        compute.execute(srcBuf: srcBuf, dstBuf: dstBuf, count: srcArray.count)
        
        // 出力バッファから数列を取得する
        let dstArray = compute.makeInt32Array(buffer: dstBuf)
        
        // 数列を出力する
        for i in dstArray {
            print("\(i)", separator: "", terminator: "" )
        }
        
        print("\n")
        
    }
}

Sample.main()
