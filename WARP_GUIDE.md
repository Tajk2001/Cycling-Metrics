# 🚀 Warp Terminal Setup Guide

## 📋 Overview

This guide shows you how to set up and use the Cycling Analysis project with Warp terminal, a modern terminal for macOS, Windows, and Linux.

## 🛠️ Initial Setup

### 1. **Navigate to Project Directory**
```bash
cd /path/to/your/cycling_analysis
```

### 2. **Activate Virtual Environment**
```bash
source venv/bin/activate
```

### 3. **Verify Installation**
```bash
python --version
pip list | grep -E "(pandas|streamlit|matplotlib)"
```

## 🎯 Quick Start Commands

### **Start Dashboard**
```bash
streamlit run dashboard.py
```

### **Run Command Line Analysis**
```bash
python cli.py --file cache/Into_the_clouds.fit --ftp 290 --name "Cyclist"
```

### **Check Cache Status**
```bash
python -c "from data_manager import CyclingDataManager; dm = CyclingDataManager(); print(dm.get_cache_info())"
```

## 🔧 Warp-Specific Features

### **1. Split Panes**
```bash
# Split horizontally
Cmd + Shift + D

# Split vertically  
Cmd + Shift + E

# Navigate between panes
Cmd + Shift + Arrow Keys
```

### **2. Workflows**
Create a workflow for cycling analysis:

```yaml
# ~/.warp/workflows/cycling_analysis.yaml
name: Cycling Analysis
shortcut: cmd+shift+c
tasks:
  - name: Start Dashboard
    command: |
      cd /path/to/your/cycling_analysis
      source venv/bin/activate
      streamlit run dashboard.py
  - name: Run Analysis
    command: |
      cd /path/to/your/cycling_analysis
      source venv/bin/activate
      python cli.py --file cache/Into_the_clouds.fit --ftp 290 --name "Cyclist"
```

### **3. Custom Commands**
Add to your shell config (`~/.zshrc` or `~/.bashrc`):

```bash
# Cycling Analysis Aliases
alias cycling-dashboard="cd /path/to/your/cycling_analysis && source venv/bin/activate && streamlit run dashboard.py"
alias cycling-analyze="cd /path/to/your/cycling_analysis && source venv/bin/activate && python cli.py"
alias cycling-cache="cd /path/to/your/cycling_analysis && source venv/bin/activate && python -c \"from data_manager import CyclingDataManager; dm = CyclingDataManager(); print(dm.get_cache_info())\""
```

## 🎯 Advanced Usage

### **1. Multi-Pane Workflow**
```bash
# Pane 1: Dashboard
streamlit run dashboard.py

# Pane 2: Command line analysis
python cli.py --file cache/Into_the_clouds.fit --ftp 290 --name "Cyclist"

# Pane 3: Monitor cache
watch -n 5 'python -c "from data_manager import CyclingDataManager; dm = CyclingDataManager(); print(dm.get_cache_info())"'
```

### **2. Batch Processing**
```bash
# Process all FIT files
for file in cache/*.fit; do
  python cli.py --file "$file" --ftp 290 --name "Cyclist"
done
```

### **3. Real-time Monitoring**
```bash
# Monitor cache directory
watch -n 2 'ls -la cache/ && echo "---" && ls -la figures/ | head -10'
```

## 🔧 Warp Configuration

### **1. Theme Setup**
```json
// ~/.warp/themes/cycling_analysis.json
{
  "name": "Cycling Analysis",
  "background": "#1a1a1a",
  "foreground": "#ffffff",
  "cursor": "#00ff00",
  "selection": "#404040",
  "black": "#000000",
  "red": "#ff0000",
  "green": "#00ff00",
  "yellow": "#ffff00",
  "blue": "#0000ff",
  "magenta": "#ff00ff",
  "cyan": "#00ffff",
  "white": "#ffffff"
}
```

### **2. Key Bindings**
```json
// ~/.warp/keybindings/cycling_analysis.json
[
  {
    "key": "cmd+shift+d",
    "command": "workbench.action.terminal.split",
    "when": "terminalFocus"
  },
  {
    "key": "cmd+shift+c",
    "command": "workbench.action.terminal.sendSequence",
    "args": {
      "text": "source venv/bin/activate && streamlit run dashboard.py\n"
    }
  }
]
```

## 🎯 Productivity Tips

### **1. Quick Commands**
```bash
# Function: Quick analysis
quick_analyze() {
  local file=$1
  local ftp=${2:-290}
  python cli.py --file "cache/$file" --ftp $ftp --name "Cyclist"
}

# Usage: quick_analyze Into_the_clouds.fit 290
```

### **2. Auto-completion**
```bash
# Add to ~/.zshrc
_cycling_analysis() {
  local cur=${COMP_WORDS[COMP_CWORD]}
  local files=$(ls cache/*.fit 2>/dev/null | xargs -n1 basename)
  COMPREPLY=( $(compgen -W "$files" -- $cur) )
}
complete -F _cycling_analysis quick_analyze
```

### **3. Status Monitoring**
```bash
# Function: Check system status
cycling_status() {
  echo "=== Cycling Analysis Status ==="
  echo "Cache files: $(ls cache/*.fit 2>/dev/null | wc -l)"
  echo "Figures: $(ls figures/*.png 2>/dev/null | wc -l)"
  echo "Virtual env: $(which python)"
  echo "================================"
}
```

## 🚨 Troubleshooting

### **1. Virtual Environment Issues**
```bash
# Problem: ModuleNotFoundError
# Solution: Ensure venv is activated
source venv/bin/activate
python -c "import pandas; print('OK')"
```

### **2. Permission Issues**
```bash
# Problem: Permission denied
# Solution: Check file permissions
ls -la venv/bin/activate
chmod +x venv/bin/activate
```

### **3. Port Conflicts**
```bash
# Problem: Port 8501 already in use
# Solution: Use different port
streamlit run dashboard.py --server.port 8502
```

### **4. Memory Issues**
```bash
# Problem: Out of memory
# Solution: Use --no-save flag
python cli.py --file cache/large_file.fit --no-save
```

## 🎯 Workflow Examples

### **Workflow 1: Daily Analysis**
```bash
# Morning routine
cycling_status
python cli.py --file cache/Into_the_clouds.fit --ftp 290 --name "Cyclist"
streamlit run dashboard.py
```

### **Workflow 2: Batch Processing**
```bash
# Process all files
for file in cache/*.fit; do
  basename=$(basename "$file" .fit)
  python cli.py --file "$file" \
    --ftp 290 \
    --name "Cyclist" \
    --output-dir "batch/$basename"
done
```

### **Workflow 3: Parameter Testing**
```bash
# Test different FTP values
for ftp in 250 270 290 310; do
  python cli.py --file cache/Into_the_clouds.fit \
    --ftp $ftp \
    --output-dir "ftp_test_$ftp"
done
```

## 🔧 Integration with Other Tools

### **1. Git Integration**
```bash
# Commit analysis results
git add figures/
git commit -m "Add analysis results for $(date +%Y-%m-%d)"
```

### **2. Backup Script**
```bash
# Backup important files
backup_cycling() {
  local backup_dir="backup_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$backup_dir"
  cp -r cache/ figures/ data/ "$backup_dir/"
  echo "Backup created: $backup_dir"
}
```

### **3. Monitoring Script**
```bash
# Monitor system resources
monitor_cycling() {
  while true; do
    clear
    echo "=== Cycling Analysis Monitor ==="
    echo "CPU: $(top -l 1 | grep "CPU usage" | awk '{print $3}')"
    echo "Memory: $(ps aux | grep python | grep -v grep | awk '{sum+=$6} END {print sum/1024 " MB"}')"
    echo "Cache: $(ls cache/*.fit 2>/dev/null | wc -l) files"
    echo "Figures: $(ls figures/*.png 2>/dev/null | wc -l) files"
    sleep 5
  done
}
```

## 📊 Performance Optimization

### **1. Use Warp's GPU Acceleration**
- Enable GPU acceleration in Warp settings
- Improves rendering performance for large outputs

### **2. Memory Management**
```bash
# Clear cache when needed
python -c "from data_manager import CyclingDataManager; dm = CyclingDataManager(); dm.clear_cache('all')"
```

### **3. Parallel Processing**
```bash
# Process multiple files in parallel (if you have multiple cores)
parallel python cli.py --file {} --ftp 290 --name "Cyclist" ::: cache/*.fit
```

---

**💡 Pro Tip**: Use Warp's split panes to run the dashboard in one pane and command-line analysis in another for maximum productivity! 