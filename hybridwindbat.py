


import numpy as np
import pandas as pd
from math import isclose
import os
from pulp import *
import matplotlib.pyplot as plt


timeLine = 8760 # hours


class BaseModel(object):

    def __init__(self,
                 fileName : str,
                 lmpFileName : str = r"hourlylmp_south.xlsx",
                 searchPath : str = r"C:\\Users\\" + str(os.getlogin()) + "\\Desktop\\batteryData",
                 battery_capacity : float = 600.0,
                 initialSOC : float = 100.0,
                 cThreshold : float = .25,
                 dThreshold : float = .95,
                 batteryPower : float = 150.0,
                 max_charge_rate : float = 150.0,
                 max_discharge_rate : float = 150.0,
                 max_daily_throughput : float = 1200.0,
                 hours : range = range(1,25),
                 firm_block_limit : float = 100.0,
                 results : list = [],
                 sellP : float = 300.0,
                 purchaseP : float = 300.0,
                 cLoss : float = .07,
                 dLoss : float = .07,
                 yearCycles : int = 365,
                 dailyCycles : int = 1,
                 firstHour : int = 0,
                 start_hour : int = 15,
                 end_hour : int = 22,
                 minSOC : int = 0) -> None:
        self.fileName, self.lmpFileName, self.searchPath, self.battery_capacity, self.initialSOC, self.cThreshold, \
            self.dThreshold, self.batteryPower, self.max_charge_rate, self.max_discharge_rate, \
                self.max_daily_throughput, self.hours, self.firm_block_limit, self.results, \
                    self.sellP, self.purchaseP, self.cLoss, self.dLoss, self.yearCycles, \
                        self.dailyCycles, self.firstHour, self.start_hour, self.end_hour, \
                             self.minSOC = fileName, lmpFileName, searchPath, \
                                battery_capacity, initialSOC, cThreshold, dThreshold, \
                                    batteryPower, max_charge_rate, max_discharge_rate, \
                                        max_daily_throughput, hours, firm_block_limit, \
                                            results, sellP, purchaseP, cLoss, dLoss, yearCycles, \
                                                dailyCycles, firstHour, start_hour, end_hour, minSOC



    def queryFile(self,
                  lmp : bool = False) -> str:
        fName = self.fileName if not lmp else self.lmpFileName
        for root, _, files in os.walk(self.searchPath):
            return os.path.join(root, fName) \
                if fName in files else None



    def getSOC(self,
               energyTransfered : float,
               asPercentage : bool = False) -> float:
        r"""
        Returns the battery state of charge (SOC).
        If asPercentage is set to True the return represents the percentage of the maximum battery capacity (%)
        else it represents the energy stored in the battery in (MWh)

        energyTransfered --> Amount of energy charging or discharging the battery resource (MWh)
        + : charging
        - : discharging
        """
        return (self.initialSOC + energyTransfered/self.battery_capacity) * 100 if asPercentage \
            else (self.initialSOC + energyTransfered/self.battery_capacity)




class ImportData(BaseModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def loadData(self) -> pd.DataFrame:
        r"Returns the load consumption dataframe in MWh"
        pass

    def solarData(self) -> pd.DataFrame:
        r"Returns the Solar Energy dataframe in MWh"
        pass

    def windData(self) -> pd.DataFrame:
        r"Returns the Wind Energy dataframe in MWh"
        windDf = pd.read_excel(self.queryFile())
        windDf = windDf[:8760]
        return windDf

    def lmpPrice(self) -> pd.Series:
        r"Returns the lmp Series (to purchase from grid) in currency/MWh"
        lmpDf = pd.read_excel(self.queryFile(lmp = True))
        lmpDf = lmpDf[:8760]
        lmpSeries = lmpDf["LMP"]
        return lmpSeries

    def sellingPrice(self) -> pd.DataFrame:
        r"Returns the selling price dataframe (to sell to the grid) in currency/MWh"
        return self.sellP

    def purchasingPrice(self) -> pd.DataFrame:
        r"Returns the electricity price (to purchase from the grid) dataframe in currency/MWh"
        return self.purchaseP



class OptimizeModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)



    def getChargeAmount(self,
                        checkDiff,
                        soc) -> float:

        if checkDiff <= self.max_charge_rate:
            if checkDiff <= self.battery_capacity - soc:
                chargeDiff = checkDiff
            else:
                chargeDiff = self.battery_capacity - soc
        else:
            if self.max_charge_rate <= self.battery_capacity - soc:
                chargeDiff = self.max_charge_rate
            else:
                chargeDiff = self.battery_capacity - soc
        return chargeDiff
      
        
        
        
        
        
        


    def getDischargeAmount(self,
                           checkDiff) -> float:
        if checkDiff <= self.max_charge_rate:
            dischargeDiff = checkDiff
        else:
            dischargeDiff = self.max_charge_rate
        return dischargeDiff
    
    
    
    def normalization(self,listsample:list)->list:
        minval=min(listsample)
        normlist=[float(i+abs(minval)) for i in listsample]
        return normlist


    def solveModel(self) -> pd.DataFrame:

        # CREATE AN INSTANCE OF THE BaseModel CLASS
        baseModel = BaseModel(fileName=r"gpwind2021gross.xlsx")

        # IMPORT THE APPROPRIATE DATA FROM THE ImportData CLASS
        importData = ImportData(**vars(baseModel))
        df = importData.windData()
        lmpSeries = importData.lmpPrice()
        sellPrice, purchasePrice = importData.sellingPrice(), importData.purchasingPrice() # currency/MWh

        ## APPLY CHANGES TO WIND DATAFRAME
        df = df["Wind"].to_frame()
        wind_profile_full = df.iloc[:, 0].tolist()
        

        ## APPLY CHANGES TO LMP SERIES
        lmpFull = lmpSeries.tolist()
        #lmpFull=[float(elem)-min(lmpFull)/(max(lmpFull)-min(lmpFull)) for elem in lmpFull]
        #normalize the lmp values 
        days=range(366)
        
       # keepsoc = []
        #mhn dineis shmasia se ayta anastash den epireazoyn
        lmplopt=[]
        lmpdict={}
        finalsortedlmp={}
        
        for year in range(1,len(wind_profile_full)//8760 +1):

            for day in range((len(wind_profile_full)//1) // len(self.hours)):
                
                # wind_profile --> List with the wind generation for every hour of each day
                wind_profile = wind_profile_full[day * len(self.hours) : (day+1) * len(self.hours)] 
                lmpL = lmpFull[day * len(self.hours) : (day+1) * len(self.hours)] 
                

                
                #oute se ayta den ta kalo kapoy telika , mhn ta dineis shmasia 
                lmplopt=lmpL[0:14]+lmpL[22:]
                for i in range(1,15):
                    lmpdict[i]=lmplopt[i-1]

                for i in range(22,24):
                    lmpdict[i]=lmplopt[i-8] # : len(peakhours+1)  generic type

                finalsortedlmp = dict(sorted(lmpdict.items(), key=lambda x: x[1]))    
               #mia logikih poy den xrhsimopoihthike 

            # CREATE A pulp MODEL
                model = LpProblem(name = "Battery Dispatch",
                                  sense = LpMaximize)


            # ASSIGN VARIABLES
            #afcharge = LpVariable.dicts(name = "Charge", 
            #af                          indices = self.hours, 
            #af                          lowBound = 0, 
            #af                          upBound = self.max_charge_rate, 
            #af                          cat = "Continuous")

                gridCharge = LpVariable.dicts(name = "gridCharge", 
                                              indices = self.hours, 
                                              lowBound = 0, 
                                              upBound = self.max_charge_rate, 
                                              cat = "Continuous")
                deficit = LpVariable.dicts(name="deficit",
                                          indices = self.hours, 
                                          lowBound = 0, 
                                          upBound = self.max_charge_rate, 
                                          cat = "Continuous")

                discharge = LpVariable.dicts("Discharge", 
                                             indices = self.hours, 
                                             lowBound = 0, 
                                             upBound = self.max_discharge_rate, 
                                             cat = "Continuous")

                total_output_poi = LpVariable.dicts("TotalOutputPOI", 
                                                    indices = self.hours, 
                                                    cat = "Continuous")

                charge_ = LpVariable.dicts("Charge or not", 
                                                   indices = self.hours, 
                                                   cat ="Binary")

                discharge_decision = LpVariable.dicts("Discharge or not", 
                                                      indices = self.hours, 
                                                      cat = "Binary")

                #soc = dict.fromkeys(list(self.hours), 0)
                
                soc = LpVariable.dicts("soc", 
                                        indices = self.hours,
                                        lowBound=0,
                                        upBound=self.battery_capacity, 
                                        cat = "Continuous")
                                       
                keepsoc=LpVariable.dicts("keepsoc",
                                         indices=days,
                                         lowBound=0,
                                         upBound=self.battery_capacity,
                                         cat="Continuous")
                
            ## DEFINE OBJECTIVE FUNCTION
            # ASSUMING THE SELLING PRICE REMAINS CONSTANT DURING THE HOURLY SIMULATION (300 currency/MWh)
                model += lpSum((total_output_poi[h] * sellPrice - gridCharge[h] * lmpL[h]-deficit[h]*lmpL[h]) for h in self.hours) # MAXIMIZE THE REVENUES OF THE HYBRID SYSTEM

            # model += lpSum((total_output_poi[h] * sellPrice) for h in self.hours)


           #af  model += lpSum((gridCharge[h] + charge[h]) for h in self.hours) <= self.dailyCycles * self.max_daily_throughput
                model += lpSum((gridCharge[h]) for h in self.hours) <= self.dailyCycles * self.max_daily_throughput
            # model += lpSum(charge[h] for h in self.hours) <= self.dailyCycles * self.max_daily_throughput
                model += lpSum(discharge[h] for h in self.hours) <= self.dailyCycles * self.max_daily_throughput


                for h in self.hours:

                #afmodel += charge[h] <= self.max_charge_rate # CHARGE AT HOUR h SHOULD BE LESS OR EQUAL THAN 200 MWh
                    #model += gridCharge[h] <= self.max_charge_rate # CHARGE AT HOUR h SHOULD BE LESS OR EQUAL THAN 200 MWh
                    model += discharge[h] <= self.max_discharge_rate # DISCHARGE AT HOUR h SHOULD BE LESS OR EQUAL THAN 200 MWh
                    #model += charge_decision[h] + discharge_decision[h] <= 1 # THE BESS SHOULD ONLY CHARGE OR DISCHARGE AT HOUR h
                    #model += gridCharge[h] <= self.battery_capacity - soc[h]
            
                    
                    

                ## ENERGY BALANCE
                # DEFINE INITIAL STATE OF CHARGE
                    """
                    if h == 0:
                            if day==0:
                                model += soc[h] == self.initialSOC
                            else:    
                        
                                model += soc[h] == keepsoc
                    else:   model+=soc[h]== soc[h-1]
                    """         
                ## DEFINE THE PEAK HOURS
                    if self.start_hour <= h <= self.end_hour:
                       # # NO CHARGING FROM GRID DURING THE ON-PEAK HOURS
                       # model += charge_decision[h] ==0
                        model+=gridCharge[h]==0
                        model += soc[h] == 0
                        
                        if wind_profile[h] >= self.firm_block_limit:
                            model += gridCharge[h] == 0
                            model+=deficit[h]==0
                            
                           # model+= discharge_decision[h]==0 
                       #af chargeDiff = self.getChargeAmount(checkDiff=wind_profile[h] - self.firm_block_limit, 
                                                          #  soc=soc[h])
                       #af model += charge[h] == chargeDiff # CHARGE THE DIFFERENCE
                            model += discharge[h] == 0 # CHARGE AND DISCHARGE SHOULD NOT TAKE PLACE AT THE SAME HOUR h
                       #af soc[h] = soc[h] + chargeDiff
                            model += total_output_poi[h] == wind_profile[h] # THE POI IS THE FIRM BLOCK LIMIT
                            model += soc[h]==soc[h-1]
                            #soc[h] == soc[h-1]
                            #model+=keepsoc==soc[h]
                    ## DISCHARGE CONDITION
                        else:
                            diff = self.firm_block_limit - wind_profile[h]
                        #af model += charge[h] == 0 # CHARGE AND DISCHARGE SHOULD NOT TAKE PLACE AT THE SAME HOUR h
                            #model+=discharge_decision[h]==1
                            
                        ## CHECK THE AMOUNT OF ENERGY STORED IN THE BESS VIA STATE OF CHARGE
                            if soc[h-1]>=diff:
                                model += gridCharge[h] == 0
                                model+=deficit[h]==0 

                                #dischargeDiff = self.getDischargeAmount(checkDiff=diff)
                                #model += discharge[h] >= dischargeDiff
                                discharge[h]>=diff and discharge[h]<=soc[h-1]
                                model += soc[h] == soc[h-1] - discharge[h]
                                model += total_output_poi[h] == self.firm_block_limit # THE POI IS THE FIRM BLOCK LIMIT
                               # model+=keespsoc==soc[h]
                        ## IF THE AMOUNT OF ENERGY STORED IN THE BESS IS NOT SUFFICIENT TO FULFILL THE FIRM BLOCK LIMIT
                            else:
                                """
                                 soc[h]=soc[h-1]
                                 if soc[h]==0:
                                    model+= discharge[h]==0
                                    model+=deficit[h]>=self.firm_block_limit

                                 else:
                                """    
                                model += gridCharge[h] == 0
                                model += discharge[h] == soc[h-1] # DISCHARGE THE AMOUNT OF ENERGY STORED IN THE BESS (soc[h] < diff)
                                model += total_output_poi[h] == discharge[h] + wind_profile[h] # THE POI IS THE SUM OF ENERGY GENERATED FROM WIND AND THE ENERGY STORED IN THE BESS ( < firm_block_limit)
                                deficit[h]>=self.firm_block_limit-total_output_poi[h]#ΑDDING THE AMOUNT OF ENERGY TO CATCH THE LIMIT 
                                deficit[h]+total_output_poi[h]>=self.firm_block_limit
                                model += soc[h] == soc[h-1] - discharge[h]
                                #model+=soc[h] == 0 # UPDATE STATE OF CHARGE (== 0 CAUSE WE DISCHARGED ALL THE AMOUNT OF ENERGY STORED IN THE BESS)
                               # model+=keepsoc==soc[h]
                            
                ## DURING THE NON-PEAK HOURS
                    else:
                        model+=deficit[h]==0 
                        model += discharge[h] == 0 # WE DO NOT NEED TO DISCHARGE ANY AMOUNT OF ENERGY DURING THE NON-PEAK HOURS
                        #model+= discharge_decision[h]==0
                        
                        gridCharge[h]>=0 and gridCharge[h]<=150 and gridCharge[h]<= self.battery_capacity-soc[h]
                        if h == 0:
                            if day==0:
                                
                                model += soc[h] == self.initialSOC+gridCharge[h]
                                
                            else:    
                                

                                model += soc[h] == keepsoc[day]+gridCharge[h]
                        else:
                            
                            

                            model+=soc[h]== soc[h-1]+gridCharge[h]
                            if h==24:

                                model+=keepsoc[day+1]==soc[h]  
                       
                       
                        #model+=soc[h]== soc[h] +gridCharge[h]
                        
                        
                          
                        
                        
                    
                    
              
                        
                        model += total_output_poi[h] == 0
                         
                                        
                            

                           

                   
                   #af  model += charge[h] == 0 # WE SHOULD NOT CHARGE THE BESS FROM THE WIND
                    
                     # WE SOULD CHARGE FROM THE GRID ALSO
                        #gChargeAmount=0
                        #if soc[h]<self.firm_block_limit:    #WHEN SOC IS LESS THAN LIMIT WE CHARGE THE MISSISNG AMOUNT 
                         #   model+=charge_decision[h]==1   
                          #  gChargeAmount = self.getChargeAmount(checkDiff=self.firm_block_limit - soc[h],soc=soc[h]) 
                        
                        #model+=gridCharge[h]*lmpL[h]>=0                                                            
                            
                        #if lmpL[h]<0: #WHEN LMPL<0 AND SOC IS MORE THAN LIMIT WE CHANRGE ON MAX RATE 
                         #   if soc[h]>=self.firm_block_limit:
                                
                          #      if gChargeAmount==0:
                           #         model+=charge_decision[h]==1
                            #        gChargeAmount+=self.max_charge_rate
                        #model += gridCharge[h] == gChargeAmount # CHARGE THE BESS FROM THE GRID
                        

                       
                       
                   

                


            ## SOLVE THE MODEL
                model.solve()



            # print("sumCharge ", lpSum(charge[k] + gridCharge[k] for k in self.hours).value())
            # print("sumDischarge ", lpSum(discharge[k] for k in self.hours).value())

            # for h in self.hours:
            #     print("h {}  charge {}   discharge {}   soc {}   wind {}   grid {}".format(h,
            #                                                                      charge[h].value(),
            #                                                                      discharge[h].value(),
            #                                                                      [soc[h].value() if (type(soc[h]) != float and type(soc[h]) != int) else soc[h]][0],
            #                                                                      wind_profile[h],
            #                                                                      gridCharge[h].value()))
            
           
                for hour in self.hours:
                    self.results.append(
                        {
                            "Year" : year,
                            "Day" : day,
                            "Hour" : hour,
                            "GridCharge(MWh)" : gridCharge[hour].value(),
                            "Deficit(MWh)" : deficit[hour].value(),
                            "Discharge(MWh)" : discharge[hour].value(),
                            "SoC(MWh)" : soc[hour].value(),
                            "KeepSoc(MWh)" : keepsoc[day].value(),
                            "Wind to POI(MWh)" : wind_profile[hour],
                            "W-F" : wind_profile[hour] - self.firm_block_limit,
                            "isPeakHour" : self.start_hour <= hour <= self.end_hour,
                            "Total Output at POI(MWh)" : total_output_poi[hour].value(),
                            "Revenues(currency)" : total_output_poi[hour].value() * sellPrice,
                            
                        }
                    )
                    


        resultFileName = r"BESSOptResultsFinal.xlsx"
        optimalDf = pd.DataFrame(self.results)

        ## CHANGE THE STATE OF CHARGE COLUMN OF THE DATAFRAME 
        # optimalDf = self.adjustCol(inptDf = optimalDf)
        optimalDf["Datetime"] = pd.to_datetime(optimalDf["Hour"], unit = "h", origin = "2023-01-01")
        optimalDf = optimalDf.set_index("Datetime")

        # optimalDf['CumulativeRevenues(currency)'] = optimalDf.groupby('Day')['Revenues(currency)'].cumsum()

        optimalDf.to_excel(self.searchPath + resultFileName, index=False)

        return optimalDf



if __name__ == "__main__":

    # CREATE AN INSTANCE OF THE BaseModel CLASS
    baseModel = BaseModel(fileName=r"gpwind2021gross.xlsx")

    # INSTANTIATE THE OptimizeModel CLASS
    opt = OptimizeModel(**vars(baseModel))

    ansDf = opt.solveModel()
    
    totalCharge, totalDischarge, totalDeficit = ( ansDf["GridCharge(MWh)"].sum()), (ansDf['Discharge(MWh)'].sum()),(ansDf["Deficit(MWh)"].sum())
    totalWind = ansDf["Wind to POI(MWh)"].sum()
    totalYield, totalRevs = ansDf["Total Output at POI(MWh)"].sum(), ansDf['Revenues(currency)'].sum()

    print("Total Energy Charge = {} (MWh)\nTotal Energy Discharge = {} (MWh)".format(format(totalCharge, ","),
                                                                                       format(totalDischarge, ",")))
    print("TotalDeficit = {} (MWh)".format(format(totalDeficit, ",")))
    print("Total Energy Wind = {} (MWh)".format(format(totalWind, ",")))

    print("Total Hybrid Energy Yield = {} (MWh)\nTotal Hybrid Total Revenues = {} (currency)".format(format(totalYield, ","),
                                                                                                       format(totalRevs, ",")))

