from pathlib import Path

path = input("Enter directory to clear -> ")
if(input("You are about to delete the contents of " + path + ". Continue? (y/n) -> ") == 'y'):
    if(input("This action cannot be undone. Continue? (y/n) -> ") == 'y'):
        P = Path(path)
        for file in P.iterdir():
            file.unlink()
        print("Directory cleared")
    else:
        print("Aborted")
else:
    print("Aborted")