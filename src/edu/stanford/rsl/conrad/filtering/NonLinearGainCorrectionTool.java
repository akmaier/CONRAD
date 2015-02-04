package edu.stanford.rsl.conrad.filtering;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.utils.CONRAD;

public class NonLinearGainCorrectionTool extends IndividualImageFilteringTool {

	/**
	 * Detector non-linear gain correction
	 * 
	 * @author Jang-Hwan Choi
	 */
	
	private static final long serialVersionUID = -5757798244081582971L;
	
	ArrayList<Double> lookUpTable;
	int start = 3800;
	int end = 4096;
	
	public ArrayList<Double> readLookUpTable() {
			
		ArrayList<Double> gainValues = new ArrayList<Double>();
		
		// initialize
		for (int i=0; i<=end; i++){
			gainValues.add(i, -1.0);			
		}
		
		
		try {
			gainValues.set(4025, 4533.01252);
			gainValues.set(4025, 4531.32776);
			gainValues.set(4025, 4529.643  );
			gainValues.set(4028, 4527.95824);
			gainValues.set(4011, 4526.27348);
			gainValues.set(4012, 4524.58872);
			gainValues.set(4010, 4522.90396);
			gainValues.set(4010, 4521.2192 );
			gainValues.set(4012, 4519.53444);
			gainValues.set(4011, 4517.84968);
			gainValues.set(4012, 4516.16492);
			gainValues.set(4015, 4514.48016);
			gainValues.set(4015, 4512.7954 );
			gainValues.set(4016, 4511.11064);
			gainValues.set(4027, 4509.42588);
			gainValues.set(4024, 4507.74112);
			gainValues.set(4028, 4506.05636);
			gainValues.set(4025, 4504.3716 );
			gainValues.set(4025, 4502.68684);
			gainValues.set(4026, 4501.00208);
			gainValues.set(4026, 4499.31732);
			gainValues.set(4027, 4497.63256);
			gainValues.set(4026, 4495.9478 );
			gainValues.set(4025, 4494.26304);
			gainValues.set(4026, 4492.57828);
			gainValues.set(4026, 4490.89352);
			gainValues.set(4025, 4489.20876);
			gainValues.set(4026, 4487.524  );
			gainValues.set(4008, 4485.83924);
			gainValues.set(4010, 4484.15448);
			gainValues.set(4009, 4482.46972);
			gainValues.set(4009, 4480.78496);
			gainValues.set(4012, 4479.1002 );
			gainValues.set(4011, 4477.41544);
			gainValues.set(4010, 4475.73068);
			gainValues.set(4010, 4474.04592);
			gainValues.set(4008, 4472.36116);
			gainValues.set(4011, 4470.6764 );
			gainValues.set(4022, 4468.99164);
			gainValues.set(4017, 4467.30688);
			gainValues.set(4019, 4465.62212);
			gainValues.set(4017, 4463.93736);
			gainValues.set(4016, 4462.2526 );
			gainValues.set(4017, 4460.56784);
			gainValues.set(4017, 4458.88308);
			gainValues.set(4019, 4457.19832);
			gainValues.set(4018, 4455.51356);
			gainValues.set(4017, 4453.8288 );
			gainValues.set(4018, 4452.14404);
			gainValues.set(4017, 4450.45928);
			gainValues.set(4016, 4448.77452);
			gainValues.set(4017, 4447.08976);
			gainValues.set(4000, 4445.405  );
			gainValues.set(4003, 4443.72024);
			gainValues.set(4005, 4442.03548);
			gainValues.set(4004, 4440.35072);
			gainValues.set(4002, 4438.66596);
			gainValues.set(4000, 4436.9812 );
			gainValues.set(3998, 4435.29644);
			gainValues.set(4001, 4433.61168);
			gainValues.set(4003, 4431.92692);
			gainValues.set(4004, 4430.24216);
			gainValues.set(4013, 4428.5574 );
			gainValues.set(4011, 4426.87264);
			gainValues.set(4011, 4425.18788);
			gainValues.set(4012, 4423.50312);
			gainValues.set(4012, 4421.81836);
			gainValues.set(4014, 4420.1336 );
			gainValues.set(4014, 4418.44884);
			gainValues.set(4015, 4416.76408);
			gainValues.set(4014, 4415.07932);
			gainValues.set(4012, 4413.39456);
			gainValues.set(4016, 4411.7098 );
			gainValues.set(4013, 4410.02504);
			gainValues.set(4012, 4408.34028);
			gainValues.set(4010, 4406.65552);
			gainValues.set(3993, 4404.97076);
			gainValues.set(3998, 4403.286  );
			gainValues.set(3997, 4401.60124);
			gainValues.set(3996, 4399.91648);
			gainValues.set(3998, 4398.23172);
			gainValues.set(3997, 4396.54696);
			gainValues.set(3997, 4394.8622 );
			gainValues.set(3996, 4393.17744);
			gainValues.set(3994, 4391.49268);
			gainValues.set(3996, 4389.80792);
			gainValues.set(4006, 4388.12316);
			gainValues.set(4004, 4386.4384 );
			gainValues.set(4003, 4384.75364);
			gainValues.set(4002, 4383.06888);
			gainValues.set(4001, 4381.38412);
			gainValues.set(4002, 4379.69936);
			gainValues.set(4003, 4378.0146 );
			gainValues.set(4003, 4376.32984);
			gainValues.set(3999, 4374.64508);
			gainValues.set(3998, 4372.96032);
			gainValues.set(4000, 4371.27556);
			gainValues.set(4000, 4369.5908 );
			gainValues.set(3999, 4367.90604);
			gainValues.set(3997, 4366.22128);
			gainValues.set(3979, 4364.53652);
			gainValues.set(3982, 4362.85176);
			gainValues.set(3982, 4361.167  );
			gainValues.set(3982, 4359.48224);
			gainValues.set(3985, 4357.79748);
			gainValues.set(3981, 4356.11272);
			gainValues.set(3979, 4354.42796);
			gainValues.set(3980, 4352.7432 );
			gainValues.set(3979, 4351.05844);
			gainValues.set(3980, 4349.37368);
			gainValues.set(3989, 4347.68892);
			gainValues.set(3988, 4346.00416);
			gainValues.set(3988, 4344.3194 );
			gainValues.set(3989, 4342.63464);
			gainValues.set(3987, 4340.94988);
			gainValues.set(3987, 4339.26512);
			gainValues.set(3983, 4337.58036);
			gainValues.set(3984, 4335.8956 );
			gainValues.set(3982, 4334.21084);
			gainValues.set(3982, 4332.52608);
			gainValues.set(3985, 4330.84132);
			gainValues.set(3983, 4329.15656);
			gainValues.set(3980, 4327.4718 );
			gainValues.set(3981, 4325.78704);
			gainValues.set(3962, 4324.10228);
			gainValues.set(3965, 4322.41752);
			gainValues.set(3965, 4320.73276);
			gainValues.set(3962, 4319.048  );
			gainValues.set(3963, 4317.36324);
			gainValues.set(3961, 4315.67848);
			gainValues.set(3961, 4313.99372);
			gainValues.set(3960, 4312.30896);
			gainValues.set(3960, 4310.6242 );
			gainValues.set(3961, 4308.93944);
			gainValues.set(3972, 4307.25468);
			gainValues.set(3967, 4305.56992);
			gainValues.set(3967, 4303.88516);
			gainValues.set(3967, 4302.2004 );
			gainValues.set(3966, 4300.51564);
			gainValues.set(3967, 4298.83088);
			gainValues.set(3967, 4297.14612);
			gainValues.set(3967, 4295.46136);
			gainValues.set(3967, 4293.7766 );
			gainValues.set(3966, 4292.09184);
			gainValues.set(3966, 4290.40708);
			gainValues.set(3964, 4288.72232);
			gainValues.set(3964, 4287.03756);
			gainValues.set(3963, 4285.3528 );
			gainValues.set(3947, 4283.66804);
			gainValues.set(3948, 4281.98328);
			gainValues.set(3947, 4280.29852);
			gainValues.set(3947, 4278.61376);
			gainValues.set(3948, 4276.929  );
			gainValues.set(3946, 4275.24424);
			gainValues.set(3946, 4273.55948);
			gainValues.set(3950, 4271.87472);
			gainValues.set(3948, 4270.18996);
			gainValues.set(3946, 4268.5052 );
			gainValues.set(3953, 4266.82044);
			gainValues.set(3954, 4265.13568);
			gainValues.set(3955, 4263.45092);
			gainValues.set(3952, 4261.76616);
			gainValues.set(3953, 4260.0814 );
			gainValues.set(3954, 4258.39664);
			gainValues.set(3954, 4256.71188);
			gainValues.set(3957, 4255.02712);
			gainValues.set(3957, 4253.34236);
			gainValues.set(3954, 4251.6576 );
			gainValues.set(3954, 4249.97284);
			gainValues.set(3952, 4248.28808);
			gainValues.set(3953, 4246.60332);
			gainValues.set(3955, 4244.91856);
			gainValues.set(3939, 4243.2338 );
			gainValues.set(3942, 4241.54904);
			gainValues.set(3940, 4239.86428);
			gainValues.set(3937, 4238.17952);
			gainValues.set(3938, 4236.49476);
			gainValues.set(3936, 4234.81   );
			gainValues.set(3936, 4233.12524);
			gainValues.set(3936, 4231.44048);
			gainValues.set(3933, 4229.75572);
			gainValues.set(3936, 4228.07096);
			gainValues.set(3947, 4226.3862 );
			gainValues.set(3944, 4224.70144);
			gainValues.set(3944, 4223.01668);
			gainValues.set(3942, 4221.33192);
			gainValues.set(3942, 4219.64716);
			gainValues.set(3942, 4217.9624 );
			gainValues.set(3940, 4216.27764);
			gainValues.set(3942, 4214.59288);
			gainValues.set(3940, 4212.90812);
			gainValues.set(3939, 4211.22336);
			gainValues.set(3942, 4209.5386 );
			gainValues.set(3941, 4207.85384);
			gainValues.set(3941, 4206.16908);
			gainValues.set(3942, 4204.48432);
			gainValues.set(3926, 4202.79956);
			gainValues.set(3927, 4201.1148 );
			gainValues.set(3926, 4199.43004);
			gainValues.set(3925, 4197.74528);
			gainValues.set(3927, 4196.06052);
			gainValues.set(3926, 4194.37576);
			gainValues.set(3924, 4192.691  );
			gainValues.set(3926, 4191.00624);
			gainValues.set(3924, 4189.32148);
			gainValues.set(3924, 4187.63672);
			gainValues.set(3934, 4185.95196);
			gainValues.set(3933, 4184.2672 );
			gainValues.set(3935, 4182.58244);
			gainValues.set(3934, 4180.89768);
			gainValues.set(3933, 4179.21292);
			gainValues.set(3934, 4177.52816);
			gainValues.set(3931, 4175.8434 );
			gainValues.set(3932, 4174.15864);
			gainValues.set(3933, 4172.47388);
			gainValues.set(3932, 4170.78912);
			gainValues.set(3934, 4169.10436);
			gainValues.set(3934, 4167.4196 );
			gainValues.set(3933, 4165.73484);
			gainValues.set(3933, 4164.05008);
			gainValues.set(3918, 4162.36532);
			gainValues.set(3920, 4160.68056);
			gainValues.set(3919, 4158.9958 );
			gainValues.set(3918, 4157.31104);
			gainValues.set(3920, 4155.62628);
			gainValues.set(3920, 4153.94152);
			gainValues.set(3918, 4152.25676);
			gainValues.set(3920, 4150.572  );
			gainValues.set(3920, 4148.88724);
			gainValues.set(3921, 4147.20248);
			gainValues.set(3933, 4145.51772);
			gainValues.set(3933, 4143.83296);
			gainValues.set(3933, 4142.1482 );
			gainValues.set(3931, 4140.46344);
			gainValues.set(3930, 4138.77868);
			gainValues.set(3932, 4137.09392);
			gainValues.set(3931, 4135.40916);
			gainValues.set(3932, 4133.7244 );
			gainValues.set(3931, 4132.03964);
			gainValues.set(3931, 4130.35488);
			gainValues.set(3932, 4128.67012);
			gainValues.set(3931, 4126.98536);
			gainValues.set(3931, 4125.3006 );
			gainValues.set(3930, 4123.61584);
			gainValues.set(3915, 4121.93108);
			gainValues.set(3918, 4120.24632);
			gainValues.set(3917, 4118.56156);
			gainValues.set(3917, 4116.8768 );
			gainValues.set(3919, 4115.19204);
			gainValues.set(3920, 4113.50728);
			gainValues.set(3916, 4111.82252);
			gainValues.set(3916, 4110.13776);
			gainValues.set(3915, 4108.453  );
			gainValues.set(3914, 4106.76824);
			gainValues.set(3927, 4105.08348);
			gainValues.set(3923, 4103.39872);
			gainValues.set(3923, 4101.71396);
			gainValues.set(3923, 4100.0292 );
			gainValues.set(3924, 4098.34444);
			gainValues.set(3926, 4096.65968);
			gainValues.set(3922, 4094.97492);
			gainValues.set(3924, 4093.29016);
			gainValues.set(3923, 4091.6054 );
			gainValues.set(3922, 4089.92064);
			gainValues.set(3924, 4088.23588);
			gainValues.set(3922, 4086.55112);
			gainValues.set(3922, 4084.86636);
			gainValues.set(3923, 4083.1816 );
			gainValues.set(3906, 4081.49684);
			gainValues.set(3908, 4079.81208);
			gainValues.set(3908, 4078.12732);
			gainValues.set(3908, 4076.44256);
			gainValues.set(3908, 4074.7578 );
			gainValues.set(3907, 4073.07304);
			gainValues.set(3907, 4071.38828);
			gainValues.set(3909, 4069.70352);
			gainValues.set(3907, 4068.01876);
			gainValues.set(3910, 4066.334  );
			gainValues.set(3919, 4064.64924);
			gainValues.set(3916, 4062.96448);
			gainValues.set(3916, 4061.27972);
			gainValues.set(3916, 4059.59496);
			gainValues.set(3913, 4057.9102 );
			gainValues.set(3911, 4056.22544);
			gainValues.set(3910, 4054.54068);
			gainValues.set(3912, 4052.85592);
			gainValues.set(3913, 4051.17116);
			gainValues.set(3912, 4049.4864 );
			gainValues.set(3914, 4047.80164);
			gainValues.set(3915, 4046.11688);
			gainValues.set(3915, 4044.43212);
			gainValues.set(3915, 4042.74736);
			gainValues.set(3898, 4041.0626 );
			gainValues.set(3901, 4039.37784);
			gainValues.set(3903, 4037.69308);
			gainValues.set(3903, 4036.00832);
			gainValues.set(3904, 4034.32356);
			gainValues.set(3902, 4032.6388 );
			gainValues.set(3901, 4030.95404);
			gainValues.set(3902, 4029.26928);
			gainValues.set(3900, 4027.58452);
			gainValues.set(3903, 4025.89976);
			gainValues.set(3907, 4024.215  );
			gainValues.set(3913, 4022.53024);
			gainValues.set(3913, 4020.84548);
			gainValues.set(3911, 4019.16072);
			gainValues.set(3910, 4017.47596);
			gainValues.set(3910, 4015.7912 );
			gainValues.set(3909, 4014.10644);
			gainValues.set(3912, 4012.42168);
			gainValues.set(3911, 4010.73692);
			gainValues.set(3911, 4009.05216);
			gainValues.set(3913, 4007.3674 );
			gainValues.set(3912, 4005.68264);
			gainValues.set(3910, 4003.99788);
			gainValues.set(3911, 4002.31312);
			gainValues.set(3896, 4000.62836);
			gainValues.set(3897, 3998.9436 );
			gainValues.set(3897, 3997.25884);
			gainValues.set(3897, 3995.57408);
			gainValues.set(3899, 3993.88932);
			gainValues.set(3897, 3992.20456);
			gainValues.set(3897, 3990.5198 );
			gainValues.set(3898, 3988.83504);
			gainValues.set(3898, 3987.15028);
			gainValues.set(3900, 3985.46552);
			gainValues.set(3909, 3983.78076);
			gainValues.set(3906, 3982.096  );
			gainValues.set(3906, 3980.41124);
			gainValues.set(3904, 3978.72648);
			gainValues.set(3904, 3977.04172);
			gainValues.set(3906, 3975.35696);
			gainValues.set(3905, 3973.6722 );
			gainValues.set(3905, 3971.98744);
			gainValues.set(3905, 3970.30268);
			gainValues.set(3905, 3968.61792);
			gainValues.set(3906, 3966.93316);
			gainValues.set(3906, 3965.2484 );
			gainValues.set(3906, 3963.56364);
			gainValues.set(3905, 3961.87888);
			gainValues.set(3889, 3960.19412);
			gainValues.set(3891, 3958.50936);
			gainValues.set(3891, 3956.8246 );
			gainValues.set(3890, 3955.13984);
			gainValues.set(3891, 3953.45508);
			gainValues.set(3891, 3951.77032);
			gainValues.set(3891, 3950.08556);
			gainValues.set(3893, 3948.4008 );
			gainValues.set(3890, 3946.71604);
			gainValues.set(3891, 3945.03128);
			gainValues.set(3901, 3943.34652);
			gainValues.set(3899, 3941.66176);
			gainValues.set(3900, 3939.977  );
			gainValues.set(3899, 3938.29224);
			gainValues.set(3899, 3936.60748);
			gainValues.set(3899, 3934.92272);
			gainValues.set(3898, 3933.23796);
			gainValues.set(3899, 3931.5532 );
			gainValues.set(3898, 3929.86844);
			gainValues.set(3897, 3928.18368);
			gainValues.set(3899, 3926.49892);
			gainValues.set(3897, 3924.81416);
			gainValues.set(3898, 3923.1294 );
			gainValues.set(3899, 3921.44464);
			gainValues.set(3885, 3919.75988);
			gainValues.set(3887, 3918.07512);
			gainValues.set(3887, 3916.39036);
			gainValues.set(3886, 3914.7056 );
			gainValues.set(3888, 3913.02084);
			gainValues.set(3886, 3911.33608);
			gainValues.set(3884, 3909.65132);
			gainValues.set(3886, 3907.96656);
			gainValues.set(3884, 3906.2818 );
			gainValues.set(3884, 3904.59704);
			gainValues.set(3887, 3902.91228);
			gainValues.set(3884, 3901.22752);
			gainValues.set(3884, 3899.54276);
			gainValues.set(3881, 3897.858  );
			gainValues.set(3881, 3896.17324);
			gainValues.set(3880, 3894.48848);
			gainValues.set(3874, 3892.80372);
			gainValues.set(3877, 3891.11896);
			gainValues.set(3878, 3889.4342 );
			gainValues.set(3874, 3887.74944);
			gainValues.set(3874, 3886.06468);
			gainValues.set(3869, 3884.37992);
			gainValues.set(3865, 3882.69516);
			gainValues.set(3867, 3881.0104 );
			gainValues.set(3861, 3879.32564);
			gainValues.set(3865, 3877.64088);
			gainValues.set(3863, 3875.95612);
			gainValues.set(3859, 3874.27136);
			gainValues.set(3861, 3872.5866 );
			gainValues.set(3858, 3870.90184);
			gainValues.set(3855, 3869.21708);
			gainValues.set(3857, 3867.53232);
			gainValues.set(3854, 3865.84756);
			gainValues.set(3853, 3864.1628 );
			gainValues.set(3851, 3862.47804);
			gainValues.set(3852, 3860.79328);
			gainValues.set(3850, 3859.10852);
			gainValues.set(3848, 3857.42376);
			gainValues.set(3844, 3855.739  );
			gainValues.set(3842, 3854.05424);
			gainValues.set(3843, 3852.36948);
			gainValues.set(3839, 3850.68472);
			gainValues.set(3837, 3848.99996);
			gainValues.set(3837, 3847.3152 );
			gainValues.set(3838, 3845.63044);
			gainValues.set(3838, 3843.94568);
			gainValues.set(3837, 3842.26092);
			gainValues.set(3834, 3840.57616);
			gainValues.set(3834, 3838.8914 );
			gainValues.set(3829, 3837.20664);
			gainValues.set(3828, 3835.52188);
			gainValues.set(3829, 3833.83712);
			gainValues.set(3825, 3832.15236);
			gainValues.set(3828, 3830.4676 );
			gainValues.set(3825, 3828.78284);
			gainValues.set(3822, 3827.09808);
			gainValues.set(3823, 3825.41332);
			gainValues.set(3819, 3823.72856);
			gainValues.set(3819, 3822.0438 );
			gainValues.set(3817, 3820.35904);
			gainValues.set(3815, 3818.67428);
			gainValues.set(3816, 3816.98952);
			gainValues.set(3811, 3815.30476);
			gainValues.set(3811, 3813.62   );
			gainValues.set(3810, 3811.93524);
			gainValues.set(3806, 3810.25048);
			gainValues.set(3807, 3808.56572);
			gainValues.set(3804, 3806.88096);
			gainValues.set(3803, 3805.1962 );
			gainValues.set(3804, 3803.51144);
			gainValues.set(3803, 3801.82668);
			gainValues.set(3799, 3800.14192);
			
		}catch(Exception e) {
			System.out.println("Exception while reading lookup table: " + e);
		}
		
		for (int i = start; i <= end; i++){
			if (gainValues.get(i) < 0){
				gainValues.set(i, gainValues.get(i-1));
			}
		}
		
		return gainValues;
	}
	
		
	public NonLinearGainCorrectionTool (){
		configured = true;
	}
	
	@Override
	public Grid2D applyToolToImage(Grid2D imageProcessor) {

		Grid2D imp = new Grid2D(imageProcessor);
		lookUpTable = readLookUpTable();
				
		for (int i = 0; i < imageProcessor.getWidth(); i++){
			for (int j = 0; j < imageProcessor.getHeight(); j++){
				int idx = Math.round(imageProcessor.getPixelValue(i, j));
				double mappedValue = imageProcessor.getPixelValue(i, j);
				
				if (idx >= start && idx <= end){	// non-linearity starting point
					mappedValue = lookUpTable.get(idx);
				}
				
				imp.putPixelValue(i, j, mappedValue);
			}
		}
		
		return imp;
	}

	@Override
	public IndividualImageFilteringTool clone() {
		IndividualImageFilteringTool clone = new NonLinearGainCorrectionTool();
		clone.configured = configured;
		return clone;
	}

	@Override
	public String getToolName() {
		return "Detector Non-linear Gain Correction Tool";
	}

	@Override
	public void configure() throws Exception {
		setConfigured(true);
	}

	@Override
	public boolean isDeviceDependent() {
		return true;
	}

	@Override
	public String getBibtexCitation() {
		return CONRAD.CONRADBibtex;
	}

	@Override
	public String getMedlineCitation() {
		return CONRAD.CONRADMedline;
	}

}

/*
 * Copyright (C) 2010-2014 - Jang Hwan Choi 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
