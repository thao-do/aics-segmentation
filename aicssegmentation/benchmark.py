from aicssegmentation.workflow import WorkflowEngine
import time
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Sequential
def main():
    batch_workflow = WorkflowEngine().get_executable_batch_workflow(
        "sec61b", 
        "/Users/sylvain.slaton/segmenter/input",
        "/Users/sylvain.slaton/segmenter/output")

    start = time.perf_counter()
    batch_workflow.process_all()
    elapsed = time.perf_counter() - start
    print(f"Program completed in {elapsed:0.5f} seconds.")

# Asyncio
async def main_async():
    batch_workflow = WorkflowEngine().get_executable_batch_workflow(
        "sec61b", 
        "/Users/sylvain.slaton/segmenter/input",
        "/Users/sylvain.slaton/segmenter/output")

    start = time.perf_counter()
    await batch_workflow.process_all_async()
    elapsed = time.perf_counter() - start
    print(f"Program completed in {elapsed:0.5f} seconds.")

def main_dask():
    batch_workflow = WorkflowEngine().get_executable_batch_workflow(
        "sec61b", 
        "/Users/sylvain.slaton/segmenter/input",
        "/Users/sylvain.slaton/segmenter/output")

    start = time.perf_counter()
    batch_workflow.process_all_dask()
    elapsed = time.perf_counter() - start
    print(f"Program completed in {elapsed:0.5f} seconds.")    


def main_dask_distributed():
    batch_workflow = WorkflowEngine().get_executable_batch_workflow(
        "sec61b", 
        "/Users/sylvain.slaton/segmenter/input",
        "/Users/sylvain.slaton/segmenter/output")

    start = time.perf_counter()
    batch_workflow.process_all_dask_distributed()
    elapsed = time.perf_counter() - start
    print(f"Program completed in {elapsed:0.5f} seconds.")    

def test():
    try:
        print("hello")
        raise TypeError("bad")
    except Exception as e:
        print("exception")
        print(str(e))

    print("done")

if __name__ == "__main__":
    asyncio.run(main_async())
    #main_dask()
    #main_dask_distributed()    