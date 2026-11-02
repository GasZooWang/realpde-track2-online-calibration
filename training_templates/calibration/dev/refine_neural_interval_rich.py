"""Extended neural interval head training."""
import refine_neural_interval as driver
from neural_interval_rich import Context,IntervalHead,FEATURE_NAMES

driver.Context=Context
driver.IntervalHead=IntervalHead
driver.FEATURE_NAMES=FEATURE_NAMES
driver.WORK=driver.ROOT/'dev/cache/neural_interval_rich'
driver.REPORT=driver.ROOT/'dev/reports/neural_interval_rich.json'

if __name__=='__main__':driver.main()
