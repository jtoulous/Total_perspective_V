from colorama import Fore, Style

def printLog(message):
    print(Style.BRIGHT + Fore.GREEN + message + Style.RESET_ALL)

def printError(message):
    print(Style.BRIGHT + Fore.RED + message + Style.RESET_ALL)

def printInfo(message):
    print(Style.BRIGHT + Fore.BLUE + message + Style.RESET_ALL)

def newCmd():
    print ('\n======================================================\n')