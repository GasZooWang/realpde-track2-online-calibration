"""Multiscale interval feature cache."""
import refine_neural_interval as driver
from cached_data import load_data
from compact_interval import Context,IntervalHead,FEATURE_NAMES

driver.Context=Context
driver.IntervalHead=IntervalHead
driver.FEATURE_NAMES=FEATURE_NAMES
driver.load_data=load_data
driver.WORK=driver.ROOT/'dev/cache/compact_interval'
driver.REPORT=driver.ROOT/'dev/reports/compact_interval.json'

if __name__=='__main__':driver.main()
