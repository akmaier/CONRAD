/*
 * Copyright (C) 2014 - Andreas Maier, Magdalena Herbst, Michael Dorner, Salah Saleh, Anja Pohan, Stefan Nottrott, Frank Schebesch 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

#define LOCAL_GROUP_XDIM 256
#define a 1
#define b 2


/* subtraction of two grids with offsets - used for gradients*/
kernel void gradient2d(global float *gridA, global float *gridB, const int xOffset, const int yOffset, const int zOffset, const int sizeX, const int sizeY, const int sizeZ, const int offsetleft)
{
	int x = get_global_id(0)+xOffset;
	int y = get_global_id(1)+yOffset;	
	
	if (( x > (sizeX+xOffset)) || (y > (sizeY+yOffset))) 
	{
		return;
	}

	for (int z = zOffset; z < sizeZ+zOffset; ++z){
		int xIdx = (x >= sizeX || x < 0) ? fmin(fmax(0.0f, x), sizeX-xOffset) : x;
		int yIdx = (y >= sizeY || y < 0) ? fmin(fmax(0.0f, y), sizeY-yOffset) : y;
		int zIdx = (z >= sizeZ || z < 0) ? fmin(fmax(0.0f, z), sizeZ-zOffset) : z;
		
		if(offsetleft == 1){
			// compute volume index for setting result grid
			int idx = (z-zOffset)*sizeX*sizeY + (y-yOffset)*sizeX + (x-xOffset);
			// compute volume indices for the subtraction
			int aIdx = zIdx*sizeX*sizeY + yIdx*sizeX + xIdx;
			int bIdx =(z-zOffset)*sizeX*sizeY + (y-yOffset)*sizeX + (x-xOffset);
			// add value to imageGrid
			gridA[idx] = gridA[aIdx] - gridB[bIdx];
		} else {
			// compute volume index for setting result grid
			int idx = (z-zOffset)*sizeX*sizeY + (y-yOffset)*sizeX + (x-xOffset);
			// compute volume indices for the subtraction
			int aIdx = (z-zOffset)*sizeX*sizeY + (y-yOffset)*sizeX + (x-xOffset);
			int bIdx = zIdx*sizeX*sizeY + yIdx*sizeX + xIdx;
			// add value to imageGrid
			gridA[idx] = gridA[aIdx] - gridB[bIdx];
		}
	}
}

/* subtraction of two grids with offsets - used for gradients*/
kernel void gradient(global float *gridA, global float *gridB, const int xOffset, const int yOffset, const int zOffset, const int sizeX, const int sizeY, const int sizeZ, const int offsetleft)
{
	int x = get_global_id(0)+xOffset;	
	
	if (x > (sizeX+xOffset)) 
	{
		return;
	}

	for(int y = yOffset;  y < sizeY+yOffset; ++y){
		for (int z = zOffset; z < sizeZ+zOffset; ++z){
			int xIdx = (x >= sizeX || x < 0) ? fmin(fmax(0.0f, x), sizeX-xOffset) : x;
			int yIdx = (y >= sizeY || y < 0) ? fmin(fmax(0.0f, y), sizeY-yOffset) : y;
			int zIdx = (z >= sizeZ || z < 0) ? fmin(fmax(0.0f, z), sizeZ-zOffset) : z;
			
			if(offsetleft == 1){
				// compute volume index for setting result grid
				int idx = (z-zOffset)*sizeX*sizeY + (y-yOffset)*sizeX + (x-xOffset);
				// compute volume indices for the subtraction
				int aIdx = zIdx*sizeX*sizeY + yIdx*sizeX + xIdx;
				int bIdx =(z-zOffset)*sizeX*sizeY + (y-yOffset)*sizeX + (x-xOffset);
				// add value to imageGrid
				gridA[idx] = gridA[aIdx] - gridB[bIdx];
			} else {
				// compute volume index for setting result grid
				int idx = (z-zOffset)*sizeX*sizeY + (y-yOffset)*sizeX + (x-xOffset);
				// compute volume indices for the subtraction
				int aIdx = (z-zOffset)*sizeX*sizeY + (y-yOffset)*sizeX + (x-xOffset);
				int bIdx = zIdx*sizeX*sizeY + yIdx*sizeX + xIdx;
				// add value to imageGrid
				gridA[idx] = gridA[aIdx] - gridB[bIdx];
			}
		}
	}
}


//Yixing Huang
kernel void compute_img_gradient(global float* grid,global float* imggradient, const int sizeX,const int sizeY,const int sizeZ)
{

int x=get_global_id(0);
int y=get_global_id(1);
//float B=10;
if((x>=sizeX) ||(y>=sizeY)||x<0||y<0)
	{
	return;
	}


for(int z=0;z<sizeZ;z++)
	{
	

			float fxyz = grid[z*sizeX*sizeY+y*sizeX+x];
			float fl = fxyz;
			float fu = fxyz;
			float ft = fxyz;
			if (x > 0)
				fl = grid[z*sizeX*sizeY+y*sizeX+x-1];
			if (y > 0)
				fu = grid[z*sizeX*sizeY+(y-1)*sizeX+x];
			if(z>0)
				ft = grid[(z-1)*sizeX*sizeY+y*sizeX+x];
			float Hdiff=fxyz-fl;
			float Vdiff=fxyz-fu;
			float Zdiff=fxyz-ft;
			imggradient[z*sizeX*sizeY+y*sizeX+x]=sqrt(Hdiff*Hdiff+Vdiff*Vdiff+Zdiff*Zdiff);
			
			}
	
}

//*****************************************************************************************************************************************************************2D case
kernel void compute_img_gradient2D(global float* grid,global float* imggradient, const int sizeX,const int sizeY)
{

int x=get_global_id(0);
int y=get_global_id(1);

if((x>=sizeX) ||(y>=sizeY)||x<0||y<0)
	{
	return;
	}

			float fxyz = grid[y*sizeX+x];
			float fl = fxyz;
			float fu = fxyz;
			
			if (x > 0)
				fl = grid[y*sizeX+x-1];
			if (y > 0)
				fu = grid[(y-1)*sizeX+x];
	
			float Hdiff=fxyz-fl;
			float Vdiff=fxyz-fu;
			
			imggradient[y*sizeX+x]=sqrt(Hdiff*Hdiff+Vdiff*Vdiff);
}


kernel void compute_Wmatrix_Update2D(global float* grid,global float* wmatrix, const float eps,const int sizeX,const int sizeY)
{
int x=get_global_id(0);
int y=get_global_id(1);

if((x>=sizeX) ||(y>=sizeY))
	{
	return;
	}
float grad=0;

	{int idx=y*sizeX+x;
	
			float fxyz = grid[idx];
			float fl = fxyz;
			float fu = fxyz;
			
			if (x > 0)
				fl = grid[idx-1];
			if (y > 0)
				fu = grid[idx-sizeX];

			float Hdiff=fxyz-fl;
			float Vdiff=fxyz-fu;
			grad=sqrt(Hdiff*Hdiff+Vdiff*Vdiff);
			wmatrix[idx]= 1.f/(grad+eps);		
	}
}


kernel void compute_wTV_gradient2D(global float* grid, global float* Wmatrix,global float* TVgradient,const int sizeX,const int sizeY)
{
float eps=0.1;
int x=get_global_id(0);
int y=get_global_id(1);

if((x>=sizeX) ||(y>=sizeY)||(x<0)||(y<0))
	{
	return;
	}
	
	float wr=0,wd=0;
	float vxy;
	float fxy = grid[y*sizeX+x];	
	float fl = fxy;
	float fr = fxy;
	float fu = fxy;
	float fd = fxy;	
	float fld =fxy;
	float fru = fxy;
	
if (x > 0)
{
	fl = grid[y*sizeX+x-1];
	}
if(x<sizeX-1)
	{
	fr=grid[y*sizeX+x+1];
	wr=Wmatrix[y*sizeX+x+1];
	}
else
	{
	wr=0.f;
	}
if (y > 0){
	fu = grid[(y-1)*sizeX+x];
	}
if(y<sizeY-1)
	{
	fd=grid[(y+1)*sizeX+x];
	wd=Wmatrix[(y+1)*sizeX+x];
	}
else
	{
	wd=0.f;
	}


if(x>0 && y<(sizeY-1)){
fld=grid[(y+1)*sizeX+x-1];
	}

if(y>0 && x<(sizeX-1)){
fru=grid[(y-1)*sizeX+x+1];
	}

vxy = Wmatrix[y*sizeX+x]*(2* fxy -  fl - fu)
						/ sqrt(eps + (fxy- fl) * (fxy - fl)+ (fxy - fu) * (fxy - fu))
						- wr* (fr - fxy)
						/ sqrt(eps + (fr - fxy) * (fr - fxy)+ (fr - fru) * (fr - fru))
						-wd* (fd - fxy)
						/ sqrt(eps + (fd - fxy) * (fd - fxy)+ (fd - fld)*(fd - fld));
						
TVgradient[y*sizeX+x]=vxy;

}

//**************************************************************************************************************************************************2D AwTV
kernel void compute_img_gradient2D_X(global float* grid,global float* imggradient, const int sizeX,const int sizeY)
{

int x=get_global_id(0);
int y=get_global_id(1);

if((x>=sizeX) ||(y>=sizeY)||x<0||y<0)
	{
	return;
	}

			float fxy = grid[y*sizeX+x];
			float fl1 = fxy,fl2=fxy,fr=fxy;
			float fu = fxy;
			
			if (x > 1){
				fl1 = grid[y*sizeX+x-1];
				fl2=grid[y*sizeX+x-2];
				}
				
			if(x<sizeX-1)
				fr=grid[y*sizeX+x+1];
			
			if (y > 0)
				fu = grid[(y-1)*sizeX+x];
	
			float Hdiff=a*fr+b*fxy-b*fl1-a*fl2;
			float Vdiff=fxy-fu;
			
			imggradient[y*sizeX+x]=sqrt(Hdiff*Hdiff+Vdiff*Vdiff);
}

kernel void compute_Wmatrix_Update2D_X(global float* grid,global float* wmatrix, const float eps,const int sizeX,const int sizeY)
{
int x=get_global_id(0);
int y=get_global_id(1);

if((x>=sizeX) ||(y>=sizeY))
	{
	return;
	}
float grad=0;

	float fxy = grid[y*sizeX+x];
			float fl1 = fxy,fl2=fxy,fr=fxy;
			float fu = fxy;
			
			if (x > 1){
				fl1 = grid[y*sizeX+x-1];
				fl2=grid[y*sizeX+x-2];
				}
				
			if(x<sizeX-1)
				fr=grid[y*sizeX+x+1];
			
			if (y > 0)
				fu = grid[(y-1)*sizeX+x];
	
			float Hdiff=a*fr+b*fxy-b*fl1-a*fl2;
			float Vdiff=fxy-fu;
			
			grad=sqrt(Hdiff*Hdiff+Vdiff*Vdiff);
			wmatrix[y*sizeX+x]=1.0f/(grad+eps);
}


//Yixing Huang
kernel void compute_wTV_gradient2D_X(global float* grid, global float* Wmatrix,global float* TVgradient,const int sizeX,const int sizeY)
{
float eps=0.1;
int x=get_global_id(0);
int y=get_global_id(1);

if((x>=sizeX) ||(y>=sizeY)||(x<0)||(y<0))
	{
	return;
	}
	



	float wr1=0,wr2=0,wl=0,wd=0,wb=0,wu=0;
	float vxy=0;
	

	if(x<3||x>sizeX-4||y<3||y>sizeY-4)
	TVgradient[y*sizeX+x]=0;
	else{	
	int idx=y*sizeX+x;
	float fxy = grid[idx];	
	float fl1 = grid[idx-1],fl2=grid[idx-2],fl3=grid[idx-3];
	float fr1 = grid[idx+1],fr2=grid[idx+2],fr3=grid[idx+3];
	float fu= grid[idx-sizeX];
	float fd= grid[idx+sizeX];
	float fl1d =grid[idx+sizeX-1],fl2d=grid[idx+sizeX-2], flu=grid[idx-sizeX-1];
	float fr1u= grid[idx-sizeX+1],fr2u=grid[idx-sizeX+2];
	float frd=grid[idx+sizeX+1];


	wr1=Wmatrix[idx+1];
	wr2=Wmatrix[idx+2];
	wl=Wmatrix[idx-1];
	wd=Wmatrix[idx+sizeX];
	
	
vxy= Wmatrix[idx]*(b*(a*fr1+b*fxy-b*fl1-a*fl2)+fxy-fu)
						/ sqrt(eps + (fxy- fu) * (fxy - fu)+ (a*fr1+b*fxy-b*fl1-a*fl2) * (a*fr1+b*fxy-b*fl1-a*fl2))
						+wd* (fxy-fd)
						/ sqrt(eps + (a*frd+b*fd-b*fl1d-a*fl2d) * (a*frd+b*fd-b*fl1d-a*fl2d)+ (fd - fxy)*(fd - fxy))
						-wr1*b*(a*fr2+b*fr1-b*fxy-a*fl1)
						/ sqrt(eps + (fr1 - fr1u) * (fr1 - fr1u)+ (a*fr2+b*fr1-b*fxy-a*fl1)* (a*fr2+b*fr1-b*fxy-a*fl1))
						-wr2*a* (a*fr3+b*fr2-b*fr1-a* fxy)
						/ sqrt(eps + (a*fr3+b*fr2-b*fr1-a* fxy) * (a*fr3+b*fr2-b*fr1-a* fxy)+ (fr2 - fr2u)*(fr2 - fr2u))
						+wl*a*(a*fxy+b*fl1-b*fl2-a*fl3)
						/ sqrt(eps + (a*fxy+b*fl1-b*fl2-a*fl3) * (a*fxy+b*fl1-b*fl2-a*fl3)+ (fl1 - flu)*(fl1 - flu))
						;
TVgradient[idx]=vxy;
}//else

}//end TV gradient largescale X



//
kernel void compute_Wmatrix_Update(global float* grid,global float* wmatrix, const float eps,const int sizeX,const int sizeY,const int sizeZ)
{

int x=get_global_id(0);
int y=get_global_id(1);

if((x>=sizeX) ||(y>=sizeY))
	{
	return;
	}
float grad=0;
for(int z=0;z<sizeZ;z++){
	{int idx=z*sizeX*sizeY+y*sizeX+x;
	
			float fxyz = grid[idx];
			float fl = fxyz;
			float fu = fxyz;
			float ft = fxyz;
			if (x > 0)
				fl = grid[idx-1];
			if (y > 0)
				fu = grid[idx-sizeX];
			if(z>0)
				ft = grid[idx-sizeX*sizeY];
			float Hdiff=fxyz-fl;
			float Vdiff=fxyz-fu;
			float Zdiff=fxyz-ft;
			grad=sqrt(Hdiff*Hdiff+Vdiff*Vdiff+Zdiff*Zdiff);
			//if(grad<0.4)
			//wmatrix[idx]=1;
			//else
			wmatrix[idx]= 1.f/(grad+eps);		
	}
}
}




//
kernel void compute_AwTV_Wmatrix_Update(global float* grid,global float* wmatrix, const float eps,const int sizeX,const int sizeY,const int sizeZ)
{

int x=get_global_id(0);
int y=get_global_id(1);
float B=100;
if((x>=sizeX) ||(y>=sizeY))
	{
	return;
	}
float grad=0;
for(int z=0;z<sizeZ;z++){
	{int idx=z*sizeX*sizeY+y*sizeX+x;
	
			float fxyz = grid[idx];
			float fl = fxyz;
			float fu = fxyz;
			float ft = fxyz;
			if (x > 0)
				fl = grid[idx-1];
			if (y > 0)
				fu = grid[idx-sizeX];
			if(z>0)
				ft = grid[idx-sizeX*sizeY];
			float Hdiff=fxyz-fl;
			float Vdiff=fxyz-fu;
			float Zdiff=fxyz-ft;
			grad=sqrt(Hdiff*Hdiff+B*Vdiff*Vdiff+Zdiff*Zdiff);
			
			wmatrix[idx]= 1.f/(grad+eps);		
	}
}
}



//
kernel void compute_Wmatrix_Update2(global float* grid,global float* wmatrix, const float eps,const int sizeX,const int sizeY,const int sizeZ)
{

int x=get_global_id(0);
int y=get_global_id(1);

if((x>=sizeX) ||(y>=sizeY))
	{
	return;
	}
float grad=0;
for(int z=0;z<sizeZ;z++){
	{int idx=z*sizeX*sizeY+y*sizeX+x;
	
			float fxyz = grid[idx];
			float fl = fxyz;
			float fu = fxyz;
			
			if (x > 0)
				fl = grid[idx-1];
			if (y > 0)
				fu = grid[idx-sizeX];
			
			float Hdiff=fxyz-fl;
			float Vdiff=fxyz-fu;
			
			grad=sqrt(Hdiff*Hdiff+Vdiff*Vdiff);
			wmatrix[idx]= 1.f/(grad+eps);		
	}
}
}

kernel void FOVmask(global float* grid,const float radius,const int sizeX,const int sizeY,const int sizeZ)
{//Set all the values outside the FOV mask as 0
int x=get_global_id(0);
int y=get_global_id(1);
float xcenter=sizeX/2.0;
float ycenter=sizeY/2.0;

	if((x>=sizeX) ||(y>=sizeY))
	{
	return;
	}
	if((x-xcenter)*(x-xcenter)+(y-ycenter)*(y-ycenter)>radius*radius){
		for(int z=0;z<sizeZ;z++){
	{
	int idx=z*sizeX*sizeY+y*sizeX+x;
	grid[idx]=0;	
	}
	}
	}
}

kernel void getwTV(global float* grid, global float* Wmatrix,global float* ZSum,const int sizeX,const int sizeY,const int sizeZ){
int x=get_global_id(0);
int y=get_global_id(1);
if((x>=sizeX) ||(y>=sizeY)||x<0||y<0)
	{
	return;
	}
float grad=0;

for(int z=0;z<sizeZ;z++)
{
int idx=z*sizeX*sizeY+y*sizeX+x;
			float fxyz = grid[idx];
			float fl = fxyz;
			float fu = fxyz;
			float ft = fxyz;
			if (x > 0)
				fl = grid[idx-1];
			if (y > 0)
				fu = grid[idx-sizeX];
			if(z>0)
				ft = grid[idx-sizeX*sizeY];
			float Hdiff=fxyz-fl;
			float Vdiff=fxyz-fu;
			float Zdiff=fxyz-ft;
			grad=sqrt(Hdiff*Hdiff+Vdiff*Vdiff+Zdiff*Zdiff);
			
			ZSum[y*sizeX+x]=ZSum[y*sizeX+x]+Wmatrix[idx]*grad;
}
}//end getwTV






kernel void getAwTV(global float* grid, global float* Wmatrix,global float* ZSum,const int sizeX,const int sizeY,const int sizeZ){
int x=get_global_id(0);
int y=get_global_id(1);
if((x>=sizeX) ||(y>=sizeY)||x<0||y<0)
	{
	return;
	}
float grad=0;
float B=100;
for(int z=0;z<sizeZ;z++)
{
int idx=z*sizeX*sizeY+y*sizeX+x;
			float fxyz = grid[idx];
			float fl = fxyz;
			float fu = fxyz;
			float ft = fxyz;
			if (x > 0)
				fl = grid[idx-1];
			if (y > 0)
				fu = grid[idx-sizeX];
			if(z>0)
				ft = grid[idx-sizeX*sizeY];
			float Hdiff=fxyz-fl;
			float Vdiff=fxyz-fu;
			float Zdiff=fxyz-ft;
			grad=sqrt(Hdiff*Hdiff+B*Vdiff*Vdiff+Zdiff*Zdiff);
			
			ZSum[y*sizeX+x]=ZSum[y*sizeX+x]+Wmatrix[idx]*grad;
}
}//end getwTV


kernel void getwTV2(global float* grid, global float* Wmatrix,global float* ZSum,const int sizeX,const int sizeY,const int sizeZ){
//without Z direction gradient
int x=get_global_id(0);
int y=get_global_id(1);
if((x>=sizeX) ||(y>=sizeY)||x<0||y<0)
	{
	return;
	}
float grad=0;
for(int z=0;z<sizeZ;z++)
{
int idx=z*sizeX*sizeY+y*sizeX+x;
			float fxyz = grid[idx];
			float fl = fxyz;
			float fu = fxyz;
			float ft = fxyz;
			if (x > 0)
				fl = grid[idx-1];
			if (y > 0)
				fu = grid[idx-sizeX];

			float Hdiff=fxyz-fl;
			float Vdiff=fxyz-fu;
			
			grad=sqrt(Hdiff*Hdiff+Vdiff*Vdiff);
			
			ZSum[y*sizeX+x]=ZSum[y*sizeX+x]+Wmatrix[idx]*grad;
}
}//end getwTV

//Yixing Huang
kernel void compute_wTV_gradient(global float* grid, global float* Wmatrix,global float* TVgradient,const int sizeX,const int sizeY,const int sizeZ)
{
float eps=0.1;
int x=get_global_id(0);
int y=get_global_id(1);

if((x>=sizeX) ||(y>=sizeY)||(x<0)||(y<0))
	{
	return;
	}
	
	float wr=0,wd=0,wb=0;
	float vxyz;

	
for(int z=0;z<sizeZ;z++)
	{int idx=z*sizeX*sizeY+y*sizeX+x;
	int Nslice=sizeX*sizeY;
	float fxyz = grid[idx];	
	float fl = fxyz;
	float fr = fxyz;
	float fu = fxyz;
	float fd = fxyz;
	float ft = fxyz;
	float fb = fxyz;
	float fld =fxyz;
	float fru = fxyz;
	float frt=  fxyz;
	float flb=  fxyz;
	float fdt=  fxyz;
	float fub=  fxyz;
if (x > 0)
{
	fl = grid[idx-1];
	}
if(x<sizeX-1)
	{
	fr=grid[idx+1];
	wr=Wmatrix[idx+1];
	}
else
	{
	wr=0.f;
	}
if (y > 0){
	fu = grid[idx-sizeX];
	}
if(y<sizeY-1)
	{
	fd=grid[idx+sizeX];
	wd=Wmatrix[idx+sizeX];
	}
else
	{
	wd=0.f;
	}

if (z > 0)
{
	ft = grid[idx-Nslice];
	}
if(z<sizeZ-1)
	{
	fb=grid[idx+Nslice];
	wb=Wmatrix[idx+Nslice];
	}
else
	{
	wb=0.f;
	}
if(x>0 && y<(sizeY-1)){
fld=grid[idx+sizeX-1];
	}

if(y>0 && x<(sizeX-1)){
fru=grid[idx-sizeX+1];
	}

if(z>0 && x<(sizeX-1)){
frt=grid[idx-Nslice+1];
	}

if(x>0 && z<(sizeZ-1)){
flb=grid[idx+Nslice-1];
	}
	
if(y>0 && z<(sizeZ-1)){
fub=grid[idx+Nslice-sizeX];
	}

if(z>0 && y<(sizeY-1)){
fdt=grid[idx-Nslice+sizeX];
	}
vxyz = Wmatrix[idx]*(3 * fxyz -  fl - fu-ft)
						/ sqrt(eps + (fxyz- fl) * (fxyz - fl)+ (fxyz - fu) * (fxyz - fu)+(fxyz-ft)*(fxyz-ft))
						- wr* (fr - fxyz)
						/ sqrt(eps + (fr - fxyz) * (fr - fxyz)+ (fr - fru) * (fr - fru)+(fr-frt)*(fr-frt))
						-wd* (fd - fxyz)
						/ sqrt(eps + (fd - fxyz) * (fd - fxyz)+ (fd - fld)*(fd - fld)+(fd-fdt)*(fd-fdt))
						-wb*(fb-fxyz)
						/sqrt(eps+(fb-flb)*(fb-flb)+(fb-fub)*(fb-fub)+(fb-fxyz)*(fb-fxyz));
TVgradient[idx]=vxyz;}
}




//Yixing Huang
kernel void compute_AwTV_gradient(global float* grid, global float* Wmatrix,global float* TVgradient,const int sizeX,const int sizeY,const int sizeZ)
{
float eps=0.1;
int x=get_global_id(0);
int y=get_global_id(1);
float B=100;
if((x>=sizeX) ||(y>=sizeY)||(x<0)||(y<0))
	{
	return;
	}
	
	float wr=0,wd=0,wb=0;
	float vxyz;

	
for(int z=0;z<sizeZ;z++)
	{
	float fxyz = grid[z*sizeX*sizeY+y*sizeX+x];	
	float fl = fxyz;
	float fr = fxyz;
	float fu = fxyz;
	float fd = fxyz;
	float ft = fxyz;
	float fb = fxyz;
	float fld =fxyz;
	float fru = fxyz;
	float frt=  fxyz;
	float flb=  fxyz;
	float fdt=  fxyz;
	float fub=  fxyz;
if (x > 0)
{
	fl = grid[z*sizeX*sizeY+y*sizeX+x-1];
	}
if(x<sizeX-1)
	{
	fr=grid[z*sizeX*sizeY+y*sizeX+x+1];
	wr=Wmatrix[z*sizeX*sizeY+y*sizeX+x+1];
	}
else
	{
	wr=0.f;
	}
if (y > 0){
	fu = grid[z*sizeX*sizeY+(y-1)*sizeX+x];
	}
if(y<sizeY-1)
	{
	fd=grid[z*sizeX*sizeY+(y+1)*sizeX+x];
	wd=Wmatrix[z*sizeX*sizeY+(y+1)*sizeX+x];
	}
else
	{
	wd=0.f;
	}

if (z > 0)
{
	ft = grid[(z-1)*sizeX*sizeY+y*sizeX+x];
	}
if(z<sizeZ-1)
	{
	fb=grid[(z+1)*sizeX*sizeY+y*sizeX+x];
	wb=Wmatrix[(z+1)*sizeX*sizeY+y*sizeX+x];
	}
else
	{
	wb=0.f;
	}
if(x>0 && y<(sizeY-1)){
fld=grid[z*sizeX*sizeY+(y+1)*sizeX+x-1];
	}

if(y>0 && x<(sizeX-1)){
fru=grid[z*sizeX*sizeY+(y-1)*sizeX+x+1];
	}

if(z>0 && x<(sizeX-1)){
frt=grid[(z-1)*sizeX*sizeY+y*sizeX+x+1];
	}

if(x>0 && z<(sizeZ-1)){
flb=grid[(z+1)*sizeX*sizeY+y*sizeX+x-1];
	}
	
if(y>0 && z<(sizeZ-1)){
fub=grid[(z+1)*sizeX*sizeY+(y-1)*sizeX+x];
	}

if(z>0 && y<(sizeY-1)){
fdt=grid[(z-1)*sizeX*sizeY+(y+1)*sizeX+x];
	}
vxyz = Wmatrix[z*sizeX*sizeY+y*sizeX+x]*((B+2)* fxyz -  fl - B*fu-ft)
						/ sqrt(eps + (fxyz- fl) * (fxyz - fl)+ B*(fxyz - fu) * (fxyz - fu)+(fxyz-ft)*(fxyz-ft))
						- wr* (fr - fxyz)
						/ sqrt(eps + (fr - fxyz) * (fr - fxyz)+ B*(fr - fru) * (fr - fru)+(fr-frt)*(fr-frt))
						-wd* B*(fd - fxyz)
						/ sqrt(eps + B*(fd - fxyz) * (fd - fxyz)+ (fd - fld)*(fd - fld)+(fd-fdt)*(fd-fdt))
						-wb*(fb-fxyz)
						/sqrt(eps+(fb-flb)*(fb-flb)+B*(fb-fub)*(fb-fub)+(fb-fxyz)*(fb-fxyz));
TVgradient[z*sizeX*sizeY+y*sizeX+x]=vxyz;}
}


//Yixing Huang
//only compute gradient in X,Y direction, without Z direction
kernel void compute_wTV_gradient2(global float* grid, global float* Wmatrix,global float* TVgradient,const int sizeX,const int sizeY,const int sizeZ)
{

float eps=0.1;
int x=get_global_id(0);
int y=get_global_id(1);

if((x>=sizeX) ||(y>=sizeY)||(x<0)||(y<0))
	{
	return;
	}
	
	float wr=0,wd=0;
	float vxyz=0;

	
for(int z=0;z<sizeZ;z++)
	{
	int idx=z*sizeX*sizeY+y*sizeX+x;
	float fxyz = grid[idx];	
	float fl = fxyz;
	float fr = fxyz;
	float fu = fxyz;
	float fd = fxyz;	
	float fld =fxyz;
	float fru = fxyz;
	
if (x > 0)
{
	fl = grid[idx-1];
	}
if(x<sizeX-1)
	{
	fr=grid[idx+1];
	wr=Wmatrix[idx+1];
	}
else
	{
	wr=0.f;
	}
if (y > 0){
	fu = grid[idx-sizeX];
	}
if(y<sizeY-1)
	{
	fd=grid[idx+sizeX];
	wd=Wmatrix[idx+sizeX];
	}
else
	{
	wd=0.f;
	}


if(x>0 && y<(sizeY-1)){
fld=grid[idx+sizeX-1];
	}

if(y>0 && x<(sizeX-1)){
fru=grid[idx-sizeX+1];
	}

vxyz = Wmatrix[idx]*(2* fxyz -  fl - fu)
						/ sqrt(eps + (fxyz- fl) * (fxyz - fl)+ (fxyz - fu) * (fxyz - fu))
						- wr* (fr - fxyz)
						/ sqrt(eps + (fr - fxyz) * (fr - fxyz)+ (fr - fru) * (fr - fru))
						-wd* (fd - fxyz)
						/ sqrt(eps + (fd - fxyz) * (fd - fxyz)+ (fd - fld)*(fd - fld));
						
TVgradient[idx]=vxyz;}

}





//
kernel void compute_adaptive_Wmatrix_Update(global float* imggradient,global float* wmatrix, const float eps,const int sizeX,const int sizeY,const int sizeZ)
{
float thres=0.2,localSum;
int hSize=4;//half of the kernel size
int x=get_global_id(0);
int y=get_global_id(1);

if((x>=sizeX) ||(y>=sizeY))
	{
	return;
	}

for(int z=0;z<sizeZ;z++){
	int idx=z*sizeX*sizeY+y*sizeX+x;
	if(x<=hSize||x>=sizeX-hSize-1||y<=hSize||y>=sizeY-hSize-1)
            wmatrix[idx]= 1.f/(imggradient[idx]+eps);	
			//wmatrix[idx]= 0;				
			else{localSum=0;
			for(int i=-hSize;i<=hSize;i++)
				for(int j=-hSize;j<=hSize;j++)
				localSum=localSum+imggradient[idx+i+j*sizeX];
				if(localSum<thres)
				wmatrix[idx]=1000000*imggradient[idx];
				//wmatrix[idx]=1;
				else
				//wmatrix[idx]= 0.f;
				wmatrix[idx]= 1.f/(imggradient[idx]+eps);
			}	
}
}


/*
kernel void compute_adaptive_Wmatrix_Update(global float* imggradient,global float* wmatrix, const float eps,const int sizeX,const int sizeY,const int sizeZ)
{
int x=get_global_id(0);
int y=get_global_id(1);

if((x>=sizeX) ||(y>=sizeY))
	{
	return;
	}

for(int z=0;z<sizeZ;z++){
	{int idx=z*sizeX*sizeY+y*sizeX+x;
			if(imggradient[idx]==1)
			wmatrix[idx]=1.f/wmatrix[idx]-eps;		
	}
}
}
*/

kernel void getwTV_adaptive(global float* grid, global float* Wmatrix,global float* ZSum,const int sizeX,const int sizeY,const int sizeZ){
int x=get_global_id(0);
int y=get_global_id(1);
if((x>=sizeX) ||(y>=sizeY)||x<0||y<0)
	{
	return;
	}
float grad=0;

for(int z=0;z<sizeZ;z++)
{
int idx=z*sizeX*sizeY+y*sizeX+x;
			float fxyz = grid[idx];
			float fl = fxyz;
			float fu = fxyz;
			float ft = fxyz;
			if (x > 0)
				fl = grid[idx-1];
			if (y > 0)
				fu = grid[idx-sizeX];
			if(z>0)
				ft = grid[idx-sizeX*sizeY];
			float Hdiff=fxyz-fl;
			float Vdiff=fxyz-fu;
			float Zdiff=fxyz-ft;
			if(Wmatrix[idx]==1)
				grad=Hdiff*Hdiff+Vdiff*Vdiff+Zdiff*Zdiff;
			else
				grad=sqrt(Hdiff*Hdiff+Vdiff*Vdiff+Zdiff*Zdiff);
			
			ZSum[y*sizeX+x]=ZSum[y*sizeX+x]+Wmatrix[idx]*grad;
}
}//end getwTV


//Yixing Huang
kernel void compute_wTV_adaptive_gradient(global float* grid, global float* Wmatrix,global float* TVgradient,const int sizeX,const int sizeY,const int sizeZ)
{
float eps=0.1;
int x=get_global_id(0);
int y=get_global_id(1);

if((x>=sizeX) ||(y>=sizeY)||(x<0)||(y<0))
	{
	return;
	}
	
	float wr=0,wd=0,wb=0;
	float vxyz=0;

	
for(int z=0;z<sizeZ;z++)
	{vxyz=0;
	int idx=z*sizeX*sizeY+y*sizeX+x;
	int Nslice=sizeX*sizeY;
	float fxyz = grid[idx];	
	float fl = fxyz;
	float fr = fxyz;
	float fu = fxyz;
	float fd = fxyz;
	float ft = fxyz;
	float fb = fxyz;
	float fld =fxyz;
	float fru = fxyz;
	float frt=  fxyz;
	float flb=  fxyz;
	float fdt=  fxyz;
	float fub=  fxyz;
if (x > 0)
{
	fl = grid[idx-1];
	}
if(x<sizeX-1)
	{
	fr=grid[idx+1];
	wr=Wmatrix[idx+1];
	}
else
	{
	wr=0.f;
	}
if (y > 0){
	fu = grid[idx-sizeX];
	}
if(y<sizeY-1)
	{
	fd=grid[idx+sizeX];
	wd=Wmatrix[idx+sizeX];
	}
else
	{
	wd=0.f;
	}

if (z > 0)
{
	ft = grid[idx-Nslice];
	}
if(z<sizeZ-1)
	{
	fb=grid[idx+Nslice];
	wb=Wmatrix[idx+Nslice];
	}
else
	{
	wb=0.f;
	}
if(x>0 && y<(sizeY-1)){
fld=grid[idx+sizeX-1];
	}

if(y>0 && x<(sizeX-1)){
fru=grid[idx-sizeX+1];
	}

if(z>0 && x<(sizeX-1)){
frt=grid[idx-Nslice+1];
	}

if(x>0 && z<(sizeZ-1)){
flb=grid[idx+Nslice-1];
	}
	
if(y>0 && z<(sizeZ-1)){
fub=grid[idx+Nslice-sizeX];
	}

if(z>0 && y<(sizeY-1)){
fdt=grid[idx-Nslice+sizeX];
	}
	if(Wmatrix[idx]==1)
	vxyz =vxyz+ Wmatrix[idx]*(3 * fxyz -  fl - fu-ft);
else
	vxyz=vxyz+Wmatrix[idx]*(3 * fxyz -  fl - fu-ft)/ sqrt(eps + (fxyz- fl) * (fxyz - fl)+ (fxyz - fu) * (fxyz - fu)+(fxyz-ft)*(fxyz-ft));

	if(wr==1)
	vxyz=vxyz- wr* (fr - fxyz);
	else
	vxyz=vxyz- wr* (fr - fxyz)
						/ sqrt(eps + (fr - fxyz) * (fr - fxyz)+ (fr - fru) * (fr - fru)+(fr-frt)*(fr-frt));
	if(wd==1)
		vxyz=vxyz-wd* (fd - fxyz);
	else
		vxyz=vxyz-wd* (fd - fxyz)
						/ sqrt(eps + (fd - fxyz) * (fd - fxyz)+ (fd - fld)*(fd - fld)+(fd-fdt)*(fd-fdt));
	if(wb==1)
		vxyz=vxyz-wb*(fb-fxyz);
	else
		vxyz=vxyz-wb*(fb-fxyz)/sqrt(eps+(fb-flb)*(fb-flb)+(fb-fub)*(fb-fub)+(fb-fxyz)*(fb-fxyz));
TVgradient[idx]=vxyz;}
}


//*************************************************************************************************************************************************wTV_Y
kernel void compute_Wmatrix_Update_Y(global float* grid,global float* wmatrix, const float eps,const int sizeX,const int sizeY,const int sizeZ)
{

int x=get_global_id(0);
int y=get_global_id(1);
float A=0.8, B=1.0;
if((x>=sizeX) ||(y>=sizeY))
	{
	return;
	}
float grad=0;
for(int z=0;z<sizeZ;z++){
	{int idx=z*sizeX*sizeY+y*sizeX+x;
	
			float fxyz = grid[idx];
			float fl = fxyz;
			float fu1 = fxyz,fu2=fxyz;
			float fd2=fxyz;
			float ft = fxyz;
			if (x > 0)
				fl = grid[idx-1];
			if (y > 2){
				fu1 = grid[idx-sizeX];
				fu2=grid[idx-2*sizeX];
				}
			if(y<(sizeY-2)){
			fd2=grid[idx+sizeX];
			}
			if(z>0)
				ft = grid[idx-sizeX*sizeY];
			float Hdiff=fxyz-fl;
			float Vdiff=b*(fxyz-fu1)+a*(fd2-fu2);
			float Zdiff=fxyz-ft;
			grad=sqrt(Hdiff*Hdiff+Vdiff*Vdiff+Zdiff*Zdiff);
			wmatrix[idx]= 1.f/((grad+eps)*10000);		
	}
}
}

kernel void getwTV_Y(global float* grid, global float* Wmatrix,global float* ZSum,const int sizeX,const int sizeY,const int sizeZ){
int x=get_global_id(0);
int y=get_global_id(1);
if((x>=sizeX) ||(y>=sizeY)||x<0||y<0)
	{
	return;
	}
float grad=0;

for(int z=0;z<sizeZ;z++)
{
int idx=z*sizeX*sizeY+y*sizeX+x;
			float fxyz = grid[idx];
			float fl = fxyz;
			float fu1 = fxyz,fu2=fxyz;
			float fd2=fxyz;
			float ft = fxyz;
			if (x > 0)
				fl = grid[idx-1];
			if (y > 2){
				fu1 = grid[idx-sizeX];
				fu2=grid[idx-2*sizeX];
				}
			if(y<(sizeY-2)){
			fd2=grid[idx+sizeX];
			}
			if(z>0)
				ft = grid[idx-sizeX*sizeY];
			float Hdiff=fxyz-fl;
			float Vdiff=b*(fxyz-fu1)+a*(fd2-fu2);
			float Zdiff=fxyz-ft;
			grad=sqrt(Hdiff*Hdiff+Vdiff*Vdiff+Zdiff*Zdiff);
			
			ZSum[y*sizeX+x]=ZSum[y*sizeX+x]+Wmatrix[idx]*grad;
}
}//end getwTV



//Yixing Huang
kernel void compute_wTV_gradient_Y(global float* grid, global float* Wmatrix,global float* TVgradient,const int sizeX,const int sizeY,const int sizeZ)
{
float eps=0.1;
int x=get_global_id(0);
int y=get_global_id(1);

if((x>=sizeX) ||(y>=sizeY)||(x<0)||(y<0))
	{
	return;
	}
	
TVgradient[(sizeZ-1)*sizeX*sizeY+y*sizeX+x]=0;//z=sizeZ-1
TVgradient[y*sizeX+x]=0;//z=0	


	float wr=0,wd1=0,wb=0,wd2=0,wu=0;
	float vxyz=0;
	
for(int z=1;z<sizeZ-1;z++)
	{
	if(x<2||x>sizeX-3||y<3||y>sizeY-4)
	TVgradient[z*sizeX*sizeY+y*sizeX+x]=0;
	else{
	
	int idx=z*sizeX*sizeY+y*sizeX+x;
	int Nslice=sizeX*sizeY;
	float fxyz = grid[idx];	
	float fl = grid[idx-1];
	float fr = grid[idx+1];
	float fu1 = grid[idx-sizeX],fu2=grid[idx-2*sizeX],fu3=grid[idx-3*sizeX];
	float fd1 = grid[idx+sizeX],fd2=grid[idx+2*sizeX],fd3=grid[idx+3*sizeX];
	float ft = grid[idx-Nslice];
	float fb = grid[idx+Nslice];
	float fld =grid[idx+sizeX-1],fld2=grid[idx+2*sizeX-1], flu=grid[idx-sizeX-1];
	float fru1 = grid[idx-sizeX+1],fru2=grid[idx-2*sizeX+1];
	float frt=  grid[idx-Nslice+1];
	float frd=grid[idx+sizeX+1];
	float flb=  grid[idx+Nslice-1];
	float fdt= grid[idx-Nslice+sizeX],fd2t=grid[idx-Nslice+2*sizeX];
	float fu1b=  grid[idx+Nslice-sizeX],fu2b=grid[idx-2*sizeX+Nslice],fdb=grid[idx+sizeX+Nslice];
	float fut=grid[idx-Nslice-sizeX];

	wr=Wmatrix[idx+1];
	wb=Wmatrix[idx+Nslice];
	wd1=Wmatrix[idx+sizeX];
	wd2=Wmatrix[idx+2*sizeX];
	wu=Wmatrix[idx-sizeX];
	
vxyz = Wmatrix[idx]*(2* fxyz -  fl -ft+b*(a*fd1+b*fxyz-b*fu1-a*fu2))
						/ sqrt(eps + (fxyz- fl) * (fxyz - fl)+ (a*fd1+b*fxyz-b*fu1-a*fu2) * (a*fd1+b*fxyz-b*fu1-a*fu2)+(fxyz-ft)*(fxyz-ft))
						+ wr* (-fr +fxyz)
						/ sqrt(eps + (fr - fxyz) * (fr - fxyz)+ (a*frd+b*fr-b*fru1-a*fru2) * (a*frd+b*fr-b*fru1-a*fru2) +(fr-frt)*(fr-frt))
						+wb*(-fb+fxyz)
						/sqrt(eps+(fb-flb)*(fb-flb)+(a*fdb+b*fb-b*fu1b-a*fu2b)*(a*fdb+b*fb-b*fu1b-a*fu2b)+(fb-fxyz)*(fb-fxyz))
						-wd1*b* (a*fd2+b*fd1- b*fxyz-a*fu1)
						/ sqrt(eps + (a*fd2+b*fd1- b*fxyz-a*fu1) * (a*fd2+b*fd1- b*fxyz-a*fu1)+ (fd1 - fld)*(fd1 - fld)+(fd1-fdt)*(fd1-fdt))
						-wd2*a* (a*fd3+b*fd2 -b*fd1-a* fxyz)
						/ sqrt(eps + (a*fd3+b*fd2 -b*fd1-a* fxyz) * (a*fd3+b*fd2 -b*fd1-a* fxyz)+ (fd2 - fld2)*(fd2 - fld2)+(fd2-fd2t)*(fd2-fd2t))
						+wu* a*(a*fxyz+b*fu1-b*fu2-a*fu3)
						/ sqrt(eps + (a*fxyz+b*fu1-b*fu2-a*fu3) * (a*fxyz+b*fu1-b*fu2-a*fu3)+ (fu1 - flu)*(fu1 - flu)+(fu1-fut)*(fu1-fut));
TVgradient[idx]=vxyz;
}//else
}//for z
}//end TV gradient largescale


//***************************************************************************************************************************************************below wTV_X

kernel void compute_Wmatrix_Update_X(global float* grid,global float* wmatrix, const float eps,const int sizeX,const int sizeY,const int sizeZ)
{

int x=get_global_id(0);
int y=get_global_id(1);

if((x>=sizeX) ||(y>=sizeY))
	{
	return;
	}
float grad=0;
for(int z=0;z<sizeZ;z++){
	{int idx=z*sizeX*sizeY+y*sizeX+x;
	
			float fxyz = grid[idx];
			float fl = fxyz;
			float fu1 = fxyz,fu2=fxyz;
			float fd2=fxyz;
			float ft = fxyz;
			if (x > 0)
				fl = grid[idx-1];
			if (y > 2){
				fu1 = grid[idx-sizeX];
				fu2=grid[idx-2*sizeX];
				}
			if(y<(sizeY-2)){
			fd2=grid[idx+sizeX];
			}
			if(z>0)
				ft = grid[idx-sizeX*sizeY];
			float Hdiff=fxyz-fl;
			float Vdiff=(fxyz-fu1)+(fd2-fu2);
			float Zdiff=fxyz-ft;
			grad=sqrt(Hdiff*Hdiff+Vdiff*Vdiff+Zdiff*Zdiff);
			wmatrix[idx]= 1.f/((grad+eps)*10000);		
	}
}
}


kernel void getwTV_X(global float* grid, global float* Wmatrix,global float* ZSum,const int sizeX,const int sizeY,const int sizeZ){
int x=get_global_id(0);
int y=get_global_id(1);
if((x>=sizeX) ||(y>=sizeY)||x<0||y<0)
	{
	return;
	}
float grad=0;

for(int z=0;z<sizeZ;z++)
{
int idx=z*sizeX*sizeY+y*sizeX+x;
			float fxyz = grid[idx];
			float fl1 = fxyz,fl2=fxyz,fr=fxyz;
			float fu = fxyz;
			float ft = fxyz;
			if (x > 2){
				fl1 = grid[idx-1];
				fl2=grid[idx-2];
				}
			if (y > 2){
				fu = grid[idx-sizeX];
				}
			if(x<(sizeX-2)){
			fr=grid[idx+1];
			}
			if(z>0)
				ft = grid[idx-sizeX*sizeY];
			float Hdiff=fxyz+fr-fl1-fl2;
			float Vdiff=fxyz-fu;
			float Zdiff=fxyz-ft;
			grad=sqrt(Hdiff*Hdiff+Vdiff*Vdiff+Zdiff*Zdiff);
			ZSum[y*sizeX+x]=ZSum[y*sizeX+x]+Wmatrix[idx]*grad;
}
}//end getwTV



//Yixing Huang
kernel void compute_wTV_gradient_X(global float* grid, global float* Wmatrix,global float* TVgradient,const int sizeX,const int sizeY,const int sizeZ)
{
float eps=0.1;
int x=get_global_id(0);
int y=get_global_id(1);

if((x>=sizeX) ||(y>=sizeY)||(x<0)||(y<0))
	{
	return;
	}
	
TVgradient[(sizeZ-1)*sizeX*sizeY+y*sizeX+x]=0;//z=sizeZ-1
TVgradient[y*sizeX+x]=0;//z=0	


	float wr1=0,wr2=0,wl=0,wd=0,wb=0,wu=0;
	float vxyz=0;
	
for(int z=1;z<sizeZ-1;z++)
	{
	if(x<3||x>sizeX-4||y<3||y>sizeY-4)
	TVgradient[z*sizeX*sizeY+y*sizeX+x]=0;
	else{
	
	int idx=z*sizeX*sizeY+y*sizeX+x;
	int Nslice=sizeX*sizeY;
	float fxyz = grid[idx];	
	float fl1 = grid[idx-1],fl2=grid[idx-2],fl3=grid[idx-3];
	float fr1 = grid[idx+1],fr2=grid[idx+2],fr3=grid[idx+3];
	float fu= grid[idx-sizeX];
	float fd= grid[idx+sizeX];
	float ft = grid[idx-Nslice];
	float fb = grid[idx+Nslice];
	float fl1d =grid[idx+sizeX-1],fl2d=grid[idx+sizeX-2], flu=grid[idx-sizeX-1];
	float fr1u= grid[idx-sizeX+1],fr2u=grid[idx-sizeX+2];
	float frd=grid[idx+sizeX+1];
	float fr1t= grid[idx-Nslice+1],fr2t=grid[idx-Nslice+2];
	float fl1b=  grid[idx+Nslice-1],fl2b=grid[idx-2+Nslice],frb=grid[idx+1+Nslice];
	float flt=grid[idx-Nslice-1];
	float fub=grid[idx-sizeX+Nslice];
	float fdt=grid[idx+sizeX-Nslice];

	wr1=Wmatrix[idx+1];
	wr2=Wmatrix[idx+2];
	wl=Wmatrix[idx-1];
	wb=Wmatrix[idx+Nslice];
	wd=Wmatrix[idx+sizeX];
	
	
vxyz = Wmatrix[idx]*(3 * fxyz+fr1- fl1-fl2-fu-ft)
						/ sqrt(eps + (fxyz- fu) * (fxyz - fu)+ (fxyz+fr1- fl1-fl2) * (fxyz+fr1 - fl1-fl2)+(fxyz-ft)*(fxyz-ft))
						+wb*(-fb+fxyz)
						/sqrt(eps+(fb-fub)*(fb-fub)+(frb+fb-fl1b-fl2b)*(frb+fb-fl1b-fl2b)+(fb-fxyz)*(fb-fxyz))
						+wd* (fxyz-fd)
						/ sqrt(eps + (frd+fd-fl1d-fl2d) * (frd+fd-fl1d-fl2d)+ (fd - fxyz)*(fd - fxyz)+(fd-fdt)*(fd-fdt))
						-wr1* (fr2+fr1-fxyz-fl1)
						/ sqrt(eps + (fr1 - fr1u) * (fr1 - fr1u)+ (fr2+fr1-fxyz-fl1) * (fr2+fr1-fxyz-fl1)+(fr1-fr1t)*(fr1-fr1t))
						-wr2* (fr3+fr2 - fxyz-fr1)
						/ sqrt(eps + (fr3+fr2 - fxyz-fr1) * (fr3+fr2 - fxyz-fr1)+ (fr2 - fr2u)*(fr2 - fr2u)+(fr2-fr2t)*(fr2-fr2t))
						+wl* (-fl2-fl3+fxyz+fl1)
						/ sqrt(eps + (-fl2-fl3+fxyz+fl1) * (-fl2-fl3+fxyz+fl1)+ (fl1 - flu)*(fl1 - flu)+(fl1-flt)*(fl1-flt))
						;
TVgradient[idx]=vxyz;
}//else
}//for z
}//end TV gradient largescale X

//*************************************************************************************************************************************************above wTV_X

kernel void UpSampling_Y(global float* DownGrid, global float* UpGrid,const int sizeX,const int sizeY,const int sizeZ){
int x=get_global_id(0);
int y=get_global_id(1);
int idx=0,idx0=0;
if((x>=sizeX) ||(y>=sizeY)||(x<0)||(y<0))
	{
	return;
	}
for(int z=0;z<sizeZ;z++){
idx=z*sizeX*sizeY*2+2*y*sizeX+x;
idx0=z*sizeX*sizeY+y*sizeX+x;
UpGrid[idx]=DownGrid[idx0];
if(y==sizeY-1)
UpGrid[idx+sizeX]=DownGrid[idx0];
else
UpGrid[idx+sizeX]=(DownGrid[idx0]+DownGrid[idx0+sizeX])/2;
}		
}

kernel void DownSampling_Y(global float* DownGrid, global float* UpGrid,const int sizeX,const int sizeY,const int sizeZ){
int x=get_global_id(0);
int y=get_global_id(1);
int idx=0,idx0=0;
if((x>=sizeX) ||(y>=sizeY)||(x<0)||(y<0))
	{
	return;
	}
for(int z=0;z<sizeZ;z++){
idx=z*sizeX*sizeY*2+2*y*sizeX+x;
idx0=z*sizeX*sizeY+y*sizeX+x;
DownGrid[idx0]=(UpGrid[idx]+UpGrid[idx+sizeX])/2;
}		
}

kernel void UpSampling_Y_even(global float* DownGrid, global float* UpGrid,const int sizeX,const int sizeY,const int sizeZ){
int x=get_global_id(0);
int y=get_global_id(1);
int idx=0,idx0=0;
if((x>=sizeX) ||(y>=sizeY)||(x<0)||(y<0))
	{
	return;
	}
for(int z=0;z<sizeZ;z++){
idx=z*sizeX*sizeY*2+2*y*sizeX+x;
idx0=z*sizeX*sizeY+y*sizeX+x;
UpGrid[idx]=DownGrid[idx0];
}		
}

kernel void UpSampling_Y_odd(global float* DownGrid, global float* UpGrid,const int sizeX,const int sizeY,const int sizeZ){
int x=get_global_id(0);
int y=get_global_id(1);
int idx=0,idx0=0;
if((x>=sizeX) ||(y>=sizeY)||(x<0)||(y<0))
	{
	return;
	}
for(int z=0;z<sizeZ;z++){
idx=z*sizeX*sizeY*2+2*y*sizeX+x+1;
idx0=z*sizeX*sizeY+y*sizeX+x;
UpGrid[idx]=DownGrid[idx0];
}		
}
