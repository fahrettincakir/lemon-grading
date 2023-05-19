import src.data as datahandling


class ImageDataGeneratorFactory:
    def create_generator(self, problem_type, datasource_type):
        raise NotImplementedError("This method should be overridden.")


class ClassificationGeneratorFactory(ImageDataGeneratorFactory):
    def create_generator(self, problem_type, datasource_type):
        if datasource_type == "coco":
            if problem_type == "binary":
                return (
                    datahandling.COCOBinaryClassificationGenerator.COCOBinaryClassificationGenerator()
                )
            elif problem_type == "multi_class":
                return (
                    datahandling.COCOMultiClassClassificationGenerator.COCOMultiClassClassificationGenerator()
                )
            elif problem_type == "multi_label":
                return (
                    datahandling.COCOMultiLabelClassificationGenerator.COCOMultiLabelClassificationGenerator()
                )
            else:
                raise ValueError("Invalid problem type.")


class SegmentationGeneratorFactory(ImageDataGeneratorFactory):
    def create_generator(self, problem_type, datasource_type):
        if datasource_type == "coco":
            if problem_type == "semantic":
                return (
                    datahandling.COCOSemanticSegmentationGenerator.COCOSemanticSegmentationGenerator()
                )
            elif problem_type == "instance":
                return datahandling.COCOInstanceSegmentationGenerator()
            else:
                raise ValueError("Invalid problem type.")
