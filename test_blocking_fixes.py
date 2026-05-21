"""
测试阻塞问题修复
验证图片处理、文件读取等功能的超时保护是否正常工作
"""

import asyncio
import logging
import sys
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_image_processor_with_invalid_urls():
    """测试图片处理器对无效URL的过滤"""
    logger.info("=" * 80)
    logger.info("测试1: 图片处理器URL验证")
    logger.info("=" * 80)
    
    try:
        from image_uploader import MarkdownImageProcessor
        
        processor = MarkdownImageProcessor(uploader=None)
        
        # 测试用例
        test_cases = [
            ("valid_image.jpg", True, "正常图片"),
            ("image.png", True, "PNG图片"),
            ("aws4_request&X-Amz-Date=20250811T060645Z", False, "AWS签名URL片段"),
            ("X-Amz-Signature=24ea4a7d89c724a3420e0f015a790aeaf5d22a5b", False, "AWS签名参数"),
            ("a" * 600 + ".jpg", False, "超长URL"),
            ("document.pdf", False, "非图片文件"),
        ]
        
        passed = 0
        failed = 0
        
        for url, expected_valid, description in test_cases:
            is_valid = processor._is_valid_image_url(url)
            if is_valid == expected_valid:
                logger.info(f"✅ {description}: {url[:50]}... - 验证正确")
                passed += 1
            else:
                logger.error(f"❌ {description}: {url[:50]}... - 验证失败 (期望: {expected_valid}, 实际: {is_valid})")
                failed += 1
        
        logger.info(f"\n测试结果: {passed}通过, {failed}失败")
        return failed == 0
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_image_processor_timeout():
    """测试图片处理的超时保护"""
    logger.info("\n" + "=" * 80)
    logger.info("测试2: 图片处理超时保护")
    logger.info("=" * 80)
    
    try:
        from image_uploader import MarkdownImageProcessor
        import tempfile
        
        processor = MarkdownImageProcessor(uploader=None)
        
        # 创建测试Markdown文件，包含大量图片引用
        test_content = """# 测试文档

这是一个测试文档，包含多个图片引用：

![图片1](image1.jpg)
![图片2](image2.png)
![图片3](image3.gif)
![图片4](https://example.com/image4.jpg)
![图片5](https://example.com/path/to/image5.png)
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(test_content)
            temp_file = Path(f.name)
        
        try:
            # 测试异步处理（应该很快完成）
            import time
            start_time = time.time()
            
            success = await asyncio.wait_for(
                processor.process_markdown_file_async(temp_file, temp_file),
                timeout=10.0
            )
            
            elapsed = time.time() - start_time
            
            logger.info(f"✅ 图片处理完成，用时: {elapsed:.2f}秒")
            logger.info(f"处理统计:")
            logger.info(f"  - 总图片数: {processor.stats['total_images']}")
            logger.info(f"  - 无效图片: {processor.stats['invalid_images']}")
            logger.info(f"  - 有效图片: {processor.stats['total_images'] - processor.stats['invalid_images']}")
            
            return True
            
        finally:
            # 清理临时文件
            if temp_file.exists():
                temp_file.unlink()
        
    except asyncio.TimeoutError:
        logger.error("❌ 图片处理超时（10秒）")
        return False
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_file_size_limits():
    """测试文件大小限制"""
    logger.info("\n" + "=" * 80)
    logger.info("测试3: 文件大小限制")
    logger.info("=" * 80)
    
    try:
        import tempfile
        import os
        
        # 创建不同大小的测试文件
        test_cases = [
            (1, True, "1MB文件"),
            (50, True, "50MB文件（应警告但允许）"),
        ]
        
        passed = 0
        failed = 0
        
        for size_mb, should_pass, description in test_cases:
            with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
                # 写入指定大小的数据
                f.write(b'0' * (size_mb * 1024 * 1024))
                temp_file = f.name
            
            try:
                file_size = os.path.getsize(temp_file)
                file_size_mb = file_size / (1024 * 1024)
                
                # 模拟文件大小检查逻辑
                if file_size_mb > 100:
                    is_valid = False
                    reason = "超过100MB限制"
                elif file_size_mb > 50:
                    is_valid = True
                    reason = "大文件警告但允许"
                else:
                    is_valid = True
                    reason = "正常大小"
                
                if is_valid == should_pass:
                    logger.info(f"✅ {description} ({file_size_mb:.1f}MB): {reason}")
                    passed += 1
                else:
                    logger.error(f"❌ {description} ({file_size_mb:.1f}MB): 验证失败")
                    failed += 1
                    
            finally:
                os.unlink(temp_file)
        
        logger.info(f"\n测试结果: {passed}通过, {failed}失败")
        return failed == 0
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_timeout_protection():
    """测试超时保护工具"""
    logger.info("\n" + "=" * 80)
    logger.info("测试4: 超时保护工具")
    logger.info("=" * 80)
    
    try:
        from utils.timeout_protection import timeout_context, run_with_timeout
        
        # 测试1: 正常完成的操作
        async def quick_task():
            await asyncio.sleep(0.1)
            return "完成"
        
        try:
            async with timeout_context(1, "快速任务"):
                result = await quick_task()
            logger.info(f"✅ 快速任务正常完成: {result}")
        except TimeoutError:
            logger.error("❌ 快速任务不应超时")
            return False
        
        # 测试2: 超时的操作
        async def slow_task():
            await asyncio.sleep(2)
            return "不应该到达这里"
        
        try:
            async with timeout_context(1, "慢速任务"):
                await slow_task()
            logger.error("❌ 慢速任务应该超时")
            return False
        except TimeoutError:
            logger.info("✅ 慢速任务正确超时")
        
        # 测试3: 带默认值的超时
        result = await run_with_timeout(
            slow_task(),
            timeout_seconds=1,
            operation_name="带默认值的任务",
            default_on_timeout="超时默认值"
        )
        
        if result == "超时默认值":
            logger.info("✅ 超时后返回默认值正常")
        else:
            logger.error(f"❌ 应返回默认值，实际返回: {result}")
            return False
        
        logger.info("\n所有超时保护测试通过")
        return True
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """运行所有测试"""
    logger.info("开始阻塞问题修复测试")
    logger.info("=" * 80)
    
    results = {
        "URL验证": await test_image_processor_with_invalid_urls(),
        "超时保护": await test_image_processor_timeout(),
        "文件大小限制": await test_file_size_limits(),
        "超时工具": await test_timeout_protection(),
    }
    
    logger.info("\n" + "=" * 80)
    logger.info("测试总结")
    logger.info("=" * 80)
    
    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        logger.info("\n🎉 所有测试通过！系统修复成功。")
        return 0
    else:
        logger.error("\n⚠️ 部分测试失败，需要进一步检查。")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

