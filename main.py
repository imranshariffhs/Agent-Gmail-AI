from logger import format_log_message, generate_trx_id, get_process_id, logger

logger.info(format_log_message(" Start Process ...."))

from gmail_agent import agent_main, cleanup_resources
from step_process import main

if __name__ == "__main__":
    trx_id = generate_trx_id()
    pid = get_process_id()
    logger.info(format_log_message("**" * 25, trx=trx_id, pid=pid))
    # Execute the main function
    try:
        logger.info(format_log_message(" Start Process ....", trx=trx_id, pid=pid))
        result = agent_main()
        if result.get("status") == "error":
            logger.error(
                format_log_message("Execution completed with issues: %s" % result["error"], trx=trx_id, pid=pid)
            )
        # logger.info("Execution completed successfully!")
    except Exception:
        logger.exception("Unexpected error in main process")
    try:
        # logger.info("Starting step process execution")
        result = main()
        # result = {"status": "success"}
        if result["status"] == "error":
            logger.error(
                format_log_message(
                    "Error occurred during step process execution: %s" % result["error"], trx=trx_id, pid=pid
                )
            )
        logger.info(format_log_message("**" * 25, trx=trx_id, pid=pid))
        # else:
        #     logger.info("Step process executed successfully")
    except Exception:
        logger.exception("Unexpected error in step process")
    logger.info(format_log_message(" End Process ....", trx=trx_id, pid=pid))

    try:
        # Run the main function
        # logger.info("Starting Gmail agent execution")
        # result = agent_main()
        result = {"status": "success"}
        if result["status"] != "success":
            logger.error(
                format_log_message("Execution completed with issues: %s" % result["error"], trx=trx_id, pid=pid)
            )
        # else:
        #     logger.info("Execution completed successfully!")
    except Exception:
        logger.exception("Unexpected error occurred")
    finally:
        cleanup_resources()
        # logger.info("Execution finished")

    # logger.info("=" * 50)
    # logger.info("Gmail agent execution completed")
    # logger.info("=" * 50)

    try:
        # logger.info("Starting step process execution")
        result = main()
        # result = {"status": "success"}
        if result["status"] == "error":
            logger.error(
                format_log_message(
                    "Error occurred during step process execution: %s" % result["error"],
                    trx=trx_id,
                    pid=pid,
                )
            )

        logger.info(format_log_message("**" * 25, trx=trx_id, pid=pid))
        # else:
        #     logger.info("Step process executed successfully")
    except Exception:
        logger.exception("Unexpected error in step process")

logger.info(format_log_message(" End Process ...."))
