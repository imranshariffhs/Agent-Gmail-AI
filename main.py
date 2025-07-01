from gmail_agent import agent_main, cleanup_resources
from step_process import main
from logger import logger


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

    try:
        # Run the main function
        logger.info("Starting Gmail agent execution")
        # result = agent_main()
        result = {"status": "success"}
        if result["status"] != "success":
            logger.error("Execution completed with issues: %s", result["error"])
        else:
            logger.info("Execution completed successfully!")
    except Exception as e:
        logger.exception("Unexpected error occurred")
    finally:
        cleanup_resources()
        logger.info("Execution finished")

    logger.info("=" * 50)
    logger.info("Gmail agent execution completed")
    logger.info("=" * 50)

    try:
        logger.info("Starting step process execution")
        result = main()
        # result = {"status": "success"}
        if result["status"] == "error":
            logger.error(
                "Error occurred during step process execution: %s", result["error"]
            )
        else:
            logger.info("Step process executed successfully")
    except Exception as e:
        logger.exception("Unexpected error in step process")
