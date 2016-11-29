package $organization$.$name;format="lower,word"$

import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize

import java.io.File

object DataIterators {
  def irisCsv(f: File): (RecordReaderDataSetIterator, DataNormalization)  = {
    val recordReader = new CSVRecordReader(0, ",")
    recordReader.initialize(new FileSplit(f))

    val labelIndex = 4;     //5 values in each row of the iris.txt CSV: 4 input features followed by an integer label (class) index. Labels are the 5th value (index 4) in each row
    val numClasses = 3;     //3 classes (types of iris flowers) in the iris data set. Classes have integer values 0, 1 or 2
    val batchSize = 50;    //Iris data set: 150 examples total.

    val iterator = new RecordReaderDataSetIterator(
      recordReader,
      batchSize,
      labelIndex,
      numClasses)
    
    val normalizer = new NormalizerStandardize()

    while (iterator.hasNext) {
      normalizer.fit(iterator.next())
    }
    iterator.reset()

    iterator.setPreProcessor(normalizer)

    (iterator, normalizer)
  }
}
