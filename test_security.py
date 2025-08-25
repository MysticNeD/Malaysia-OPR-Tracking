# test_security_report.py
import requests
import os

BASE_URL = "https://malaysia-opr-tracking.onrender.com"
REPORT_FILE = "security_report.txt"

def write_report(lines):
    with open(REPORT_FILE, "w", encoding="utf-8") as f:   # 加上 encoding="utf-8"
        f.write("\n".join(lines))
    print(f"\n📄 安全报告已生成: {REPORT_FILE}")


def read_env_mode():
    env_file = ".env"
    mode = "dev"
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                if line.strip().startswith("ENV="):
                    mode = line.strip().split("=")[1]
                elif line.strip().startswith("DEBUG="):
                    debug_val = line.strip().split("=")[1].lower()
                    if debug_val == "false":
                        mode = "prod"
    return mode

def test_protected_endpoint(report_lines):
    report_lines.append("1️⃣ 测试受保护接口（Auth）...")
    try:
        r = requests.get(f"{BASE_URL}/secure-data")
        if r.status_code == 401:
            report_lines.append("✅ 正确：未授权访问被拒绝")
        else:
            report_lines.append(f"⚠️ 注意：返回状态码 {r.status_code}")
    except Exception as e:
        report_lines.append(f"❌ 访问接口失败: {e}")

def test_cors(report_lines):
    report_lines.append("2️⃣ 测试 CORS...")
    headers = {"Origin": "http://evil.com"}
    try:
        r = requests.options(f"{BASE_URL}/", headers=headers)
        if "access-control-allow-origin" not in r.headers:
            report_lines.append("✅ 正确：未允许的 Origin 被拦截")
        else:
            report_lines.append(f"⚠️ 注意：CORS header 存在 {r.headers['access-control-allow-origin']}")
    except Exception as e:
        report_lines.append(f"❌ CORS 测试失败: {e}")

def test_static_protection(report_lines):
    report_lines.append("3️⃣ 测试静态/敏感文件保护...")
    sensitive_files = [".env", "data/mydb.db", "logs/app.log"]
    for f in sensitive_files:
        try:
            r = requests.get(f"{BASE_URL}/{f}")
            if r.status_code in [403, 404]:
                report_lines.append(f"✅ 正确：{f} 访问被拒绝")
            else:
                report_lines.append(f"⚠️ 注意：{f} 可以访问，状态码 {r.status_code}")
        except Exception as e:
            report_lines.append(f"❌ {f} 测试失败: {e}")

def test_logs_masking(report_lines):
    report_lines.append("4️⃣ 测试日志敏感信息掩码...")
    log_file = "logs/app.log"
    if not os.path.exists(log_file):
        report_lines.append("⚠️ 日志文件不存在，跳过")
        return
    with open(log_file, "r") as f:
        content = f.read()
        if "password" in content.lower() or "secret" in content.lower():
            report_lines.append("⚠️ 日志包含敏感信息")
        else:
            report_lines.append("✅ 日志未包含敏感信息")

if __name__ == "__main__":
    print("==== 开始安全性自动测试 ====")
    mode = read_env_mode()
    print(f"🔹 当前环境模式: {mode.upper()}\n")
    
    report_lines = [f"安全性测试报告 - 当前环境: {mode.upper()}"]
    test_protected_endpoint(report_lines)
    test_cors(report_lines)
    test_static_protection(report_lines)
    test_logs_masking(report_lines)
    
    write_report(report_lines)
    print("==== 安全性测试完成 ====")
