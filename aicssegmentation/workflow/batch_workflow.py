import numpy as np
import logging
import asyncio

from dask.distributed import Client, progress, as_completed
from typing import Union
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter
from pathlib import Path
from aicssegmentation.util.filesystem import FileSystemUtilities
from aicssegmentation.exceptions import ArgumentNullError
from .workflow import Workflow
from .workflow_definition import WorkflowDefinition

log = logging.getLogger(__name__)

SUPPORTED_FILE_EXTENSIONS = [".tiff", ".tif", ".czi"]


class BatchWorkflow:
    """
    Represents a batch of workflows to process.
    This class provides the functionality to run batches of workflows using multiple image inputs from a input directory
    according to the steps defined in its WorkflowDefinition.
    """

    def __init__(
        self,
        workflow_definition: WorkflowDefinition,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        channel_index: int = 0,
    ):
        if workflow_definition is None:
            raise ArgumentNullError("workflow_definition")
        if input_dir is None:
            raise ArgumentNullError("input_dir")
        if output_dir is None:
            raise ArgumentNullError("output_dir")

        self._workflow_definition = workflow_definition
        self._input_dir = Path(input_dir)

        if not self._input_dir.exists():
            raise ValueError("The input directory does not exist")

        self._output_dir = Path(output_dir)
        self._channel_index = channel_index
        self._files_count: int = 0
        self._failed_files: int = 0
        self._log_path: Path = self._output_dir / "log.txt"

        # Create the output directory at output_dir if it does not exist already
        if not self._output_dir.exists():
            FileSystemUtilities.create_directory(self._output_dir)

    @property
    def input_dir(self) -> Path:
        return self._input_dir

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    def is_valid_image(self, image_path: Path) -> bool:
        """
        Check if file at a given image_path and is a valid image type we support.

        Params:
            image_path (Path): image to check

        Returns:
            (bool): True if file has a supported file extension.
        """
        if not image_path.exists():
            return False
        if image_path.suffix.lower() in SUPPORTED_FILE_EXTENSIONS:
            return True
        else:
            return False

    async def process_file_async(self, file: Path):
        read_image = AICSImage(file)

        try:
            # read and format image in the way we expect
            image_from_path = self.format_image_to_3d(read_image)
            # Run workflow on image
            workflow = Workflow(self._workflow_definition, image_from_path)
            result = await workflow.execute_all_async()
            with OmeTiffWriter(self._output_dir.joinpath(file.name), overwrite_file=True) as w:
                w.save(data=self.convert_bool_to_uint8(result), dimension_order="ZYX")

        except Exception as e:
            # Handle failures during workflow execution/save
            self._failed_files += 1
            #self._write_to_log_file(f"FAILED: {file}, ERROR: {e}\n")

    def process_file(self, file: Path):
        read_image = AICSImage(file)

        try:
            # read and format image in the way we expect
            image_from_path = self.format_image_to_3d(read_image)
            # Run workflow on image
            workflow = Workflow(self._workflow_definition, image_from_path)
            result = workflow.execute_all()
            with OmeTiffWriter(self._output_dir / f"{file.stem}.segmentation.tiff", overwrite_file=True) as w:
                w.save(data=self.convert_bool_to_uint8(result), dimension_order="ZYX")    
            print("*** FILE SAVED ***")        
        except Exception as e:
            print(f"*** EXCEPTION *** {e}")
            # Handle failures during workflow execution/save
            self._failed_files += 1
            #self._write_to_log_file(f"FAILED: {file}, ERROR: {e}\n")        
        return file.name

    def process_all_dask(self):
        import dask
        log.info("START::process_all_dask")
        self._write_to_log_file("Log for batch processing run\n")

        files = [f for f in self._input_dir.glob("**/*") if f.is_file]
        # Currently will save files in same format as they are in the input path
        tasks = list()
        for f in files:
            log.info(f"File={f.name}")
            self._files_count += 1
            if self.is_valid_image(f):
                tasks.append(dask.delayed(self.process_file)(f))
            else:
                self._failed_files += 1
                self._write_to_log_file(f"FAILED: {f}, ERROR: Unsupported Image Type {f.suffix}\n")

        log.info(f"Waiting on {len(tasks)} tasks...")

        #client = Client()
        #x = dask.persist(*tasks)
        #progress(x)
        
        dask.compute(*tasks)
        #dask.compute(x)
        
        
        self._write_log_file_summary()
        log.info("END::process_all_dask")

    def process_all_dask_distributed(self):
        import dask
        log.info("START::process_all_dask_distributed")
        self._write_to_log_file("Log for batch processing run\n")

        files = [f for f in self._input_dir.glob("**/*") if f.is_file]
        # Currently will save files in same format as they are in the input path
        #tasks = list()
        client = Client()
        futures = list()
        for f in files:
            log.info(f"File={f.name}")
            self._files_count += 1
            if self.is_valid_image(f):
                futures.append(client.submit(self.process_file, f))
            else:
                self._failed_files += 1
                self._write_to_log_file(f"FAILED: {f}, ERROR: Unsupported Image Type {f.suffix}\n")

        log.info(f"Waiting on {len(futures)} tasks...")

        #progress(futures)
        for future, result in as_completed(futures, with_results=True):
            print(f"**** TASK COMPLETED *** {result}")

        client.close()
        self._write_log_file_summary()
        log.info("END::process_all_dask")

    async def process_all_async(self):
        """
        Process all images in the input_dir with the workflow_definition used to set up the BatchWorkflow

        Params:
            none

        Returns:
            none
        """
        print("START::process_all_async")
        log.info("START::process_all_async")
        self._write_to_log_file("Log for batch processing run\n")

        files = [f for f in self._input_dir.glob("**/*") if f.is_file]
        # Currently will save files in same format as they are in the input path
        tasks = list()
        for f in files:
            log.info(f"File={f.name}")
            self._files_count += 1
            if self.is_valid_image(f):
                tasks.append(self.process_file_async(f))
            else:
                self._failed_files += 1
                self._write_to_log_file(f"FAILED: {f}, ERROR: Unsupported Image Type {f.suffix}\n")

        log.info(f"Waiting on {len(tasks)} tasks...")
        await asyncio.gather(*tasks)
        self._write_log_file_summary()

    def process_all(self):
        """
        Process all images in the input_dir with the workflow_definition used to set up the BatchWorkflow

        Params:
            none

        Returns:
            none
        """

        self._write_to_log_file("Log for batch processing run\n")

        files = [f for f in self._input_dir.glob("**/*") if f.is_file]
        # Currently will save files in same format as they are in the input path
        for f in files:
            self._files_count += 1
            if self.is_valid_image(f):
                read_image = AICSImage(f)

                try:
                    # read and format image in the way we expect
                    image_from_path = self.format_image_to_3d(read_image)
                    # Run workflow on image
                    workflow = Workflow(self._workflow_definition, image_from_path)
                    result = workflow.execute_all()
                    with OmeTiffWriter(self._output_dir.joinpath(f.name), overwrite_file=True) as w:
                        w.save(data=self.convert_bool_to_uint8(result), dimension_order="ZYX")

                except Exception as e:
                    # Handle failures during workflow execution/save
                    self._failed_files += 1
                    self._write_to_log_file(f"FAILED: {f}, ERROR: {e}\n")
            else:
                self._failed_files += 1
                self._write_to_log_file(f"FAILED: {f}, ERROR: Unsupported Image Type {f.suffix}\n")

        self._write_log_file_summary()

    def _write_log_file_summary(self):
        """
        Write a log file to the output folder.
        """
        if self._files_count == 0:
            report = "There were no files to process in the input directory"
        else:

            files_processed = self._files_count - self._failed_files
            report = f"{files_processed}/{self._files_count} files were processed.\n"
        self._write_to_log_file(report + "\n")

    def format_image_to_3d(self, image: AICSImage) -> np.ndarray:
        """
        Format images in the way that aics-segmention expects for most workflows (3d, zyx)

        Params:
            image_path (AICSImage): image to format

        Returns:
            np.ndarray: segment-able image for aics-segmentation
        """
        if image.size_s > 1:
            raise ValueError("Multi-Scene images are unsupported")

        if image.size_t > 1:
            raise ValueError("Timelapse images are unsupported.")

        if image.size_c > 1:
            return image.get_image_data("ZYX", C=self._channel_index)

        return image.get_image_data("ZYX")

    def convert_bool_to_uint8(self, image: np.ndarray):
        """
        Format segmented images to uint8 to save via AICSImage

        Params:
            image (np.ndarray): segmented image

        Returns:
            np.ndarray: image converted to uint8 for saving
        """
        image = image.astype(np.uint8)
        image[image > 0] = 255
        return image

    def _write_to_log_file(self, text: str):
        with open(self._log_path, "a") as writer:
            writer.write(text)
