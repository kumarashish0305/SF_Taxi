dbutils.fs.mkdirs("/mnt/cabspotting")
#Copy file from CSV download link to databricks envrionment  
#Important first data is copied from open source data link to a temporary mount

%sh wget  wget https://raw.githubusercontent.com/PDXostc/rvi_big_data/master/cabspottingdata.tar.gz -O cabspottingdata.tar.gz -O /tmp/cabspottingdata.tar.gz

#Move data from Temporay folder to Crime_la_data folder
dbutils.fs.mv("file:///tmp/cabspottingdata.tar.gz", "/mnt/cabspot/cabspottingdata.tar.gz")

#Check all the text files which are part of the tar file
import tarfile
tar = tarfile.open("/dbfs/mnt/cabspot/cabspottingdata.tar.gz", "r:gz")
for tarinfo in tar:
    print(tarinfo.name, "is", tarinfo.size, "bytes in size and is", end="")
    if tarinfo.isreg():
        print(" a regular file.")
    elif tarinfo.isdir():
        print(" a directory.")
    else:
        print(" something else.")
tar.close()

#Next Command Cell
import tarfile
tar = tarfile.open("/dbfs/mnt/cabspot/cabspottingdata.tar.gz")
tar.extractall("/dbfs/mnt/")
tar.close()
dbutils.fs.rm("/mnt/cabspottingdata/README",True)

#Next Command Cell
#Create a List with all filenames
from os import listdir
from os.path import isfile, join
filenames = [f for f in listdir("/dbfs/mnt/cabspottingdata") if isfile(join("/dbfs/mnt/cabspottingdata", f))]

#Remove common words new_ and .txt
filenames = ' '.join(filenames).replace('new_','').split()
filenames = ' '.join(filenames).replace('.txt','').split()

#Next Command Cell
# Create one combined file
with open('/dbfs/mnt/taxall.txt', 'w') as outfile:
    for fname in filenames:
        with open("/dbfs/mnt/cabspottingdata/new_" + fname + ".txt") as infile:
            for line in infile:
                new_line = " ".join([fname, line])
                outfile.write(new_line)
                
#Next command cell
#Create schema for text file
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, LongType, DateType, StringType
customSchema = StructType([
    StructField("TaxiID", StringType(), True), 
    StructField("latitude", DoubleType(), True),        
    StructField("longitude", DoubleType(), True),
    StructField("occupancy", IntegerType(), True),
    StructField("time", LongType(), True)
])

taxi_df = sqlContext.read.format("com.databricks.spark.csv").option("header", "false").option("delimiter", " ").schema(customSchema).load("/mnt/taxall.txt")

#Count total rows for spark dataframe : 11 million rows+ount total rows for spark dataframe : 11 million rows+
taxi_df.count()
#1million+ records

#Display spark dataframe
display(taxi_df)

#Print the Schema / Metadata for the dataset
taxi_df.printSchema()

#Make a directory for parquet format
dbutils.fs.mkdirs("/mnt/parquet/sf_taxi")
#Save as Parquet file
taxi_df.write.mode("overwrite").format("parquet").save("/mnt/parquet/sf_taxi")

#Let us physically see where the files are 
#Important Dbutils is available in R , Python and SQL
dbutils.fs.rm("/mnt/cabspotting",True)
dbutils.fs.rm("/mnt/taxall.txt",True)
dbutils.fs.rm("/mnt/cabspottingdata",True)
#Alternatively use %fs to use spark code but comments cannot be added into the cell referring to another programming language in python script
#%fs or File system uilities are not available in R and SQL

#Observe files stored
%fs ls /mnt/






