from gmail_agent import agent_main
from step_process import main


if __name__ == "__main__":
    # Execute the main function
    try:
        result = agent_main()
        if result["status"] == "error":
            print("\nError occurred during execution:")
            print(result["error"])
        else:
            print("Gmail agent executed successfully")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")

    print("************************************************")
    print("Gmail agent executed successfully")
    print("************************************************")

    try:
        result = main()
        if result["status"] == "error":
            print("\nError occurred during execution:")
            print(result["error"])
        else:
            print("Step process executed successfully")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")




            