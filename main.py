"""
RAG 文档预处理主程序
整合三阶段流水线，提供命令行接口
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import (
    validate_config,
    get_config_summary,
    OutputConfig,
    LogConfig,
    PerformanceConfig
)
from processing_stages.stage1_baseline import Stage1Baseline
from processing_stages.stage2_clean_map import Stage2CleanMap
from processing_stages.stage3_refine_locate import Stage3RefineLocate


# 配置日志
def setup_logging(log_level: str = None, log_file: str = None):
    """配置日志系统"""
    level = getattr(logging, log_level or LogConfig.LOG_LEVEL)
    log_format = LogConfig.LOG_FORMAT

    handlers = [logging.StreamHandler()]

    if log_file or LogConfig.LOG_FILE:
        file_handler = logging.FileHandler(
            log_file or LogConfig.LOG_FILE,
            encoding='utf-8'
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=handlers
    )


logger = logging.getLogger(__name__)


class RAGPreprocessor:
    """RAG 文档预处理器主类"""

    def __init__(self):
        self.stage1 = Stage1Baseline()
        self.stage2 = Stage2CleanMap()
        self.stage3 = Stage3RefineLocate()

    def process_file(self, input_file: str, output_dir: str = None) -> Dict[str, Any]:
        """
        处理单个 Markdown 文件

        Args:
            input_file: 输入文件路径
            output_dir: 输出目录路径

        Returns:
            处理结果字典
        """
        logger.info("=" * 80)
        logger.info(f"开始处理文件: {input_file}")
        logger.info("=" * 80)

        start_time = datetime.now()

        # 1. 读取文件
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                markdown_text = f.read()
            logger.info(f"✅ 文件读取成功: {len(markdown_text)} 字符")
        except Exception as e:
            logger.error(f"❌ 文件读取失败: {e}")
            raise

        # 2. 阶段1: 基线建立与粗切
        logger.info("\n" + "=" * 80)
        stage1_result = self.stage1.process(markdown_text)

        # 保存阶段1中间结果
        if OutputConfig.SAVE_INTERMEDIATE_RESULTS:
            self._save_intermediate_result(
                stage1_result,
                output_dir,
                OutputConfig.STAGE1_OUTPUT,
                serialize_tokens=False  # 不序列化 Token 列表
            )

        # 3. 阶段2: 智能清洗与 Token 映射
        logger.info("\n" + "=" * 80)
        if PerformanceConfig.ENABLE_ASYNC:
            import asyncio
            stage2_result = asyncio.run(self.stage2.async_process(stage1_result))
        else:
            stage2_result = self.stage2.process(stage1_result)

        # 保存阶段2中间结果
        if OutputConfig.SAVE_INTERMEDIATE_RESULTS:
            self._save_intermediate_result(
                stage2_result,
                output_dir,
                OutputConfig.STAGE2_OUTPUT,
                serialize_tokens=False
            )

        # 4. 阶段3: 精细切分与最终定位
        logger.info("\n" + "=" * 80)
        stage3_result = self.stage3.process(stage2_result)

        # 5. 保存最终结果
        output_file = self._save_final_result(
            stage3_result,
            output_dir,
            input_file
        )

        # 6. 生成处理报告
        elapsed = datetime.now() - start_time
        report = self._generate_report(
            input_file,
            output_file,
            stage1_result,
            stage2_result,
            stage3_result,
            elapsed
        )

        logger.info("\n" + "=" * 80)
        logger.info("处理完成!")
        logger.info("=" * 80)
        logger.info(f"\n{report}")

        return {
            "input_file": input_file,
            "output_file": output_file,
            "final_chunks": stage3_result["final_chunks"],
            "statistics": stage3_result["statistics"],
            "processing_time": elapsed.total_seconds(),
            "report": report
        }

    def _save_intermediate_result(
        self,
        result: Dict[str, Any],
        output_dir: str,
        filename: str,
        serialize_tokens: bool = True
    ):
        """保存中间结果"""
        output_dir = output_dir or OutputConfig.OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, filename)

        # 移除不可序列化的字段
        serializable_result = self._make_serializable(result, serialize_tokens)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(
                serializable_result,
                f,
                indent=OutputConfig.JSON_INDENT,
                ensure_ascii=OutputConfig.JSON_ENSURE_ASCII
            )

        logger.info(f"💾 中间结果已保存: {output_path}")

    def _save_final_result(
        self,
        result: Dict[str, Any],
        output_dir: str,
        input_file: str
    ) -> str:
        """保存最终结果"""
        output_dir = output_dir or OutputConfig.OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)

        # 生成输出文件名
        input_name = Path(input_file).stem
        output_filename = f"{input_name}_final_chunks.json"
        output_path = os.path.join(output_dir, output_filename)

        # 读取原始文本以计算字符位置
        with open(input_file, 'r', encoding='utf-8') as f:
            original_text = f.read()

        # 初始化tokenizer来计算字符位置
        from tokenizer.tokenizer_client import get_tokenizer
        tokenizer = get_tokenizer()
        base_tokens = tokenizer.encode(original_text)

        logger.info(f"📍 计算字符位置 (原文共 {len(base_tokens)} tokens, {len(original_text)} 字符)...")

        # 为每个chunk计算字符位置
        chunks_with_char_pos = []
        for i, chunk in enumerate(result["final_chunks"]):
            token_start = chunk["token_start"]
            token_end = chunk["token_end"]

            # 从开头到token_start的文本
            if token_start > 0:
                prefix_tokens = base_tokens[:token_start]
                prefix_text = tokenizer.decode(prefix_tokens)
                char_start = len(prefix_text)
            else:
                char_start = 0

            # 从开头到token_end的文本
            if token_end > 0 and token_end <= len(base_tokens):
                prefix_tokens = base_tokens[:token_end]
                prefix_text = tokenizer.decode(prefix_tokens)
                char_end = len(prefix_text)
            else:
                char_end = len(original_text)

            chunks_with_char_pos.append({
                "chunk_id": i + 1,
                "content": chunk["content"],
                "token_start": token_start,
                "token_end": token_end,
                "token_count": chunk["token_count"],
                "char_start": char_start,
                "char_end": char_end,
                "user_tag": chunk["user_tag"],
                "content_tags": chunk["content_tags"],
                "is_atomic": chunk["is_atomic"],
                "atomic_type": chunk["atomic_type"]
            })

        # 准备输出数据
        output_data = {
            "metadata": {
                "source_file": input_file,
                "processed_at": datetime.now().isoformat(),
                "total_chunks": len(result["final_chunks"]),
                "statistics": result["statistics"]
            },
            "chunks": chunks_with_char_pos
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(
                output_data,
                f,
                indent=OutputConfig.JSON_INDENT,
                ensure_ascii=OutputConfig.JSON_ENSURE_ASCII
            )

        logger.info(f"💾 最终结果已保存: {output_path}")
        return output_path

    def _make_serializable(
        self,
        data: Any,
        serialize_tokens: bool = True
    ) -> Any:
        """使数据可序列化"""
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                # 跳过 Token 列表（如果不序列化）
                if not serialize_tokens and key in ['base_tokens', 'tokens']:
                    result[key] = f"<{len(value)} tokens>"
                else:
                    result[key] = self._make_serializable(value, serialize_tokens)
            return result
        elif isinstance(data, list):
            return [self._make_serializable(item, serialize_tokens) for item in data]
        else:
            return data

    def _generate_report(
        self,
        input_file: str,
        output_file: str,
        stage1_result: Dict[str, Any],
        stage2_result: Dict[str, Any],
        stage3_result: Dict[str, Any],
        elapsed
    ) -> str:
        """生成处理报告"""
        s1_stats = stage1_result["statistics"]
        s2_stats = stage2_result["statistics"]
        s3_stats = stage3_result["statistics"]

        report = f"""
╔══════════════════════════════════════════════════════════════╗
║              RAG 文档预处理报告                                ║
╠══════════════════════════════════════════════════════════════╣
║ 输入文件: {Path(input_file).name:<45} ║
║ 输出文件: {Path(output_file).name:<45} ║
║ 处理时间: {elapsed.total_seconds():.2f} 秒{' ' * 43}║
╠══════════════════════════════════════════════════════════════╣
║ 阶段1 - 基线建立与粗切                                          ║
║   原始 Tokens: {s1_stats['total_tokens']:<46} ║
║   原始字符数: {s1_stats['total_chars']:<47} ║
║   Mid-Chunks: {s1_stats['mid_chunk_count']:<48} ║
║   平均 Tokens: {s1_stats['avg_chunk_tokens']:.1f}{' ' * 45}║
╠══════════════════════════════════════════════════════════════╣
║ 阶段2 - 智能清洗与Token映射                                     ║
║   Clean-Chunks: {s2_stats['clean_chunk_count']:<45} ║
║   清洗后 Tokens: {s2_stats['total_cleaned_tokens']:<44} ║
║   平均 Tokens: {s2_stats['avg_chunk_tokens']:.1f}{' ' * 45}║
╠══════════════════════════════════════════════════════════════╣
║ 阶段3 - 精细切分与最终定位                                      ║
║   最终块数量: {s3_stats['total_chunks']:<47} ║
║   平均 Tokens: {s3_stats['avg_tokens']:.1f}{' ' * 45}║
║   Token 范围: {s3_stats['min_tokens']}-{s3_stats['max_tokens']}{' ' * 45}║
║   ATOMIC 块: {s3_stats['atomic_chunks']:<48} ║
║   验证通过率: {s3_stats['validation_pass_rate']:.1%}{' ' * 45}║
╚══════════════════════════════════════════════════════════════╝
"""
        return report


def main():
    """主函数：命令行入口"""
    parser = argparse.ArgumentParser(
        description='RAG 文档预处理工具 - 智能清洗与语义切分',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py input.md
  python main.py input.md -o ./output
  python main.py input.md --log-level DEBUG
        """
    )

    parser.add_argument(
        'input_file',
        help='输入的 Markdown 文件路径'
    )

    parser.add_argument(
        '-o', '--output-dir',
        help=f'输出目录路径 (默认: {OutputConfig.OUTPUT_DIR})',
        default=None
    )

    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help=f'日志级别 (默认: {LogConfig.LOG_LEVEL})',
        default=None
    )

    parser.add_argument(
        '--log-file',
        help=f'日志文件路径 (默认: {LogConfig.LOG_FILE})',
        default=None
    )

    parser.add_argument(
        '--validate-config',
        action='store_true',
        help='验证配置并退出'
    )

    args = parser.parse_args()

    # 配置日志
    setup_logging(args.log_level, args.log_file)

    try:
        # 验证配置
        logger.info("验证配置...")
        validate_config()
        logger.info("✅ 配置验证通过")

        # 显示配置摘要
        config_summary = get_config_summary()
        logger.info(f"\n配置摘要: {json.dumps(config_summary, indent=2, ensure_ascii=False)}")

        if args.validate_config:
            logger.info("✅ 配置验证完成，退出")
            return 0

        # 检查输入文件
        if not os.path.exists(args.input_file):
            logger.error(f"❌ 输入文件不存在: {args.input_file}")
            return 1

        # 创建处理器并执行
        preprocessor = RAGPreprocessor()
        result = preprocessor.process_file(
            args.input_file,
            args.output_dir
        )

        logger.info(f"\n✅ 处理完成! 输出文件: {result['output_file']}")
        return 0

    except KeyboardInterrupt:
        logger.warning("\n⚠️ 用户中断")
        return 130

    except Exception as e:
        logger.error(f"\n❌ 处理失败: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
