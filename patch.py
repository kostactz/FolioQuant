import re

with open('src/app/assets/dash_clientside.js', 'r') as f:
    content = f.read()

def patch_func(func_name, return_val, content):
    pattern = r'(' + func_name + r':\s*function\s*\([^)]*\)\s*\{)(?!\s*if \(Date\.now)'
    replacement = r'\1\n        if (Date.now() - window.__folioquant_mount_time < 1500) return ' + return_val + r';'
    return re.sub(pattern, replacement, content)

content = patch_func('update_depth_chart', 'window.dash_clientside.no_update', content)
content = patch_func('update_analyst_metrics', 'Array(5).fill(window.dash_clientside.no_update)', content)
content = patch_func('update_ofi_chart', 'window.dash_clientside.no_update', content)
content = patch_func('update_metrics_chart', 'window.dash_clientside.no_update', content)
content = patch_func('update_execution_chart', 'window.dash_clientside.no_update', content)

with open('src/app/assets/dash_clientside.js', 'w') as f:
    f.write(content)
