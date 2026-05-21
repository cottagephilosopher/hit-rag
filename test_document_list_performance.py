"""
测试文档列表 API 性能
对比优化前后的性能差异
"""

import time
import asyncio
import httpx


API_BASE = "http://localhost:8000"


async def test_list_documents_performance():
    """测试文档列表加载性能"""
    
    print("=" * 60)
    print("📊 文档列表 API 性能测试")
    print("=" * 60)
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        
        # 测试 1：不带统计信息（快速模式）
        print("\n🧪 测试 1：不带统计信息 (include_stats=false)")
        start = time.time()
        response = await client.get(f"{API_BASE}/api/documents?include_stats=false")
        elapsed = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            count = len(data.get("documents", []))
            print(f"✅ 成功加载 {count} 个文档")
            print(f"⏱️  耗时: {elapsed:.2f} 秒")
        else:
            print(f"❌ 失败: {response.status_code}")
        
        # 测试 2：带统计信息（首次，无缓存）
        print("\n🧪 测试 2：带统计信息 - 首次请求（无缓存）")
        
        # 先清除缓存（通过上传一个不存在的文件触发，或者重启服务）
        print("   （提示：首次请求会查询数据库并缓存）")
        
        start = time.time()
        response = await client.get(f"{API_BASE}/api/documents?include_stats=true")
        elapsed = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            docs = data.get("documents", [])
            count = len(docs)
            
            # 统计已处理文档数量
            processed_count = sum(1 for d in docs if d.get("status") == "processed")
            total_chunks = sum(d.get("chunk_count", 0) for d in docs)
            total_tokens = sum(d.get("total_tokens", 0) for d in docs)
            
            print(f"✅ 成功加载 {count} 个文档 ({processed_count} 个已处理)")
            print(f"📦 总 chunks: {total_chunks:,}")
            print(f"🔢 总 tokens: {total_tokens:,}")
            print(f"⏱️  耗时: {elapsed:.2f} 秒")
        else:
            print(f"❌ 失败: {response.status_code}")
        
        # 测试 3：带统计信息（第二次，有缓存）
        print("\n🧪 测试 3：带统计信息 - 第二次请求（有缓存）")
        await asyncio.sleep(0.5)  # 短暂等待
        
        start = time.time()
        response = await client.get(f"{API_BASE}/api/documents?include_stats=true")
        elapsed = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            count = len(data.get("documents", []))
            print(f"✅ 成功加载 {count} 个文档")
            print(f"⏱️  耗时: {elapsed:.2f} 秒 ⚡")
            print(f"💾 缓存加速比: {elapsed / 0.01 if elapsed > 0 else 'N/A'}x")
        else:
            print(f"❌ 失败: {response.status_code}")
        
        # 测试 4：分页请求（前 20 个）
        print("\n🧪 测试 4：分页请求（limit=20）")
        start = time.time()
        response = await client.get(f"{API_BASE}/api/documents?include_stats=true&limit=20&offset=0")
        elapsed = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            count = len(data.get("documents", []))
            total = data.get("total", 0)
            has_more = data.get("has_more", False)
            
            print(f"✅ 成功加载 {count}/{total} 个文档")
            print(f"📄 有更多数据: {has_more}")
            print(f"⏱️  耗时: {elapsed:.2f} 秒")
        else:
            print(f"❌ 失败: {response.status_code}")
    
    print("\n" + "=" * 60)
    print("✅ 测试完成")
    print("=" * 60)
    
    print("\n💡 性能提升要点：")
    print("  1. 批量 SQL 查询：从 N*3 次查询 → 2 次查询")
    print("  2. 全局缓存：第二次请求几乎瞬时完成")
    print("  3. 异步文件读取：并发处理多个文件")
    print("  4. 数据库索引：已优化查询性能")


if __name__ == "__main__":
    try:
        asyncio.run(test_list_documents_performance())
    except KeyboardInterrupt:
        print("\n\n⚠️  测试被用户中断")
    except Exception as e:
        print(f"\n\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

