import subprocess

def uninstall_all_packages():
    result = subprocess.run(['pip', 'freeze'], stdout=subprocess.PIPE)
    installed_packages = result.stdout.decode('utf-8').split('\n')
    for package in installed_packages:
        if package:
            subprocess.run(['pip', 'uninstall', '-y', package.split('==')[0]])

if __name__ == "__main__":
    uninstall_all_packages()
