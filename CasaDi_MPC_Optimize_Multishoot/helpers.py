import yaml
import os

def load_config(config_file):
    """
    加载yaml配置文件
    Args:
        config_file: yaml配置文件的路径
    Returns:
        包含配置信息的字典
    """
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"错误：找不到配置文件 {config_file}")
        raise
    except yaml.YAMLError as e:
        print(f"错误：YAML文件格式错误 - {e}")
        raise
    except Exception as e:
        print(f"错误：加载配置文件时发生未知错误 - {e}")
        raise

def get_config_path(config_file):
    """
    获取配置文件的完整路径
    Args:
        config_file: 配置文件名
    Returns:
        配置文件的完整路径
    """
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建配置文件的完整路径
    config_path = os.path.join(current_dir, config_file)
    return config_path

def validate_config(config):
    """
    验证配置文件的必要参数
    Args:
        config: 配置字典
    Returns:
        bool: 配置是否有效
    """
    required_sections = ['mpc_params', 'velocity_constraints', 'vehicle_params', 'dynamics_constraints']
    
    for section in required_sections:
        if section not in config:
            print(f"错误：配置文件中缺少必要的部分 '{section}'")
            return False
    
    return True

def print_config(config):
    """
    打印配置信息（用于调试）
    Args:
        config: 配置字典
    """
    print("当前配置信息：")
    for section, values in config.items():
        print(f"\n{section}:")
        for key, value in values.items():
            print(f"  {key}: {value}")

