from Process.pipeline import Pipeline
from Utils.logger import Logger


def main():
    # 필요한 경우 사용자 정의 모듈 경로 추가
    logger_instance = Logger('main', 'pipeline.log', when='midnight', interval=1, backup_count=7)
    main_logger = logger_instance.get_logger()

    try:
        pipeline = Pipeline(main_logger)
        pipeline.run()
    except Exception as e:
        main_logger.exception("An error occurred during the pipeline execution.")
        raise



if __name__ == '__main__': 
    main()
