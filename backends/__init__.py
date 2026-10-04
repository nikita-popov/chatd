elif prefix == "deepseek":
    from backends.deepseek import DeepSeekBackend
    return DeepSeekBackend()
