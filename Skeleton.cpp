/* Azért 1 fájlban van minden kód, mert a feladatbeadó portál így kérte
   és nem volt kedvem beadás után foglalkozni a projekttel. */

#include "framework.h"
const char * const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0

	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

const char * const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec4 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = color;	// computed color is the color of the primitive
	}
)";

const char * vertexSourceForBackground = R"(
	#version 330
    precision highp float;

	uniform mat4 MVP;			// Model-View-Projection matrix in row-major format

	layout(location = 0) in vec2 vertexPosition;	// Attrib Array 0
	layout(location = 1) in vec2 vertexUV;			// Attrib Array 1

	out vec2 texCoord;								// output attribute

	void main() {
		texCoord = vertexUV;														// copy texture coordinates
		gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1) * MVP; 		// transform to clipping space
	}
)";

const char * fragmentSourceForBackground = R"(
	#version 330
    precision highp float;
	
	in vec2 texCoord;			// variable input: interpolated texture coordinates
	out vec4 outColor;		// output that goes to the raster memory as told by glBindFragDataLocation

bool isHill(vec2 Coord){
	 float tension = 0.85f;
	 vec4 leftside;
	 vec4 actual;
	 vec4 rightside;
	 vec4 twoToTheRight;
	    vec4 controlPoints[8]=vec4[8](
		vec4(-0.1, 0.6, 0, 0),
		vec4(0.2, 0.7, 0, 0),
		vec4(0.4,0.45, 0, 0),
		vec4(0.6,0.7, 0, 0),
		vec4(0.8,0.65, 0, 0),
		vec4(1,0.4, 0, 0),
		vec4(1.2,0.6, 0, 0),
		vec4(1.4, 0.6, 0, 0)
	);	
		for(int i = 1; i < 7; i++){
			if(controlPoints[i].x > Coord.x){
				leftside = controlPoints[i-2];
				actual = controlPoints[i-1];
				rightside = controlPoints[i];
				twoToTheRight = controlPoints[i+1];
				break;
			}
		}
		 vec4 v0 = ((rightside - actual) / (rightside.x - actual.x) +
						(actual - leftside) / (actual.x - leftside.x))  *  (1 - tension ) / 2.0f;
		
		 vec4 v1 = ((twoToTheRight - rightside) / (twoToTheRight.x - rightside.x) + 
						(rightside - actual) / (rightside.x - actual.x))  * (1 - tension) / 2.0f;
		 float Dt = rightside.x - actual.x;
		 float dt = Coord.x - actual.x;
		 vec4 a3 = ((actual - rightside) * 2 / pow(Dt, 3) + (v1 + v0) / pow(Dt, 2));
		 vec4 a2 = ((rightside - actual) * 3 / pow(Dt, 2) - (v1 + v0 * 2) / Dt);
		 vec4 a1 = v0;
		 vec4 a0 = actual;
		float y = ( a3* pow(dt, 3) + a2 * pow(dt,2) + a1 * dt + a0).y;
		return y > Coord.y;
	}

	void main() {
		if (isHill(texCoord)) {
			outColor = vec4(131/255.0f, 204/255.0f, 211/255.0f, 1.0f); 
		} else {
			  vec4 skyTop = vec4(158/255.0f, 236/255.0f, 1.0f, 1.0f);
			  vec4 skyBottom = vec4(45.0f/255.0f,212.0f/255.0f, 1.0f, 1.0f); 
			  float blend = texCoord.y;
			 outColor = (1-blend)*skyTop + blend*skyTop;
		
		}
	}
)";
bool FollowingBike = false;
enum  Orientation {
	left = -1,
	right = 1
};


const float PI = 3.1415926535897f;
const float appliedZoom = 0.7f;
GPUProgram gpuProgram; 
GPUProgram backGroundMaker;
unsigned int vao;	  

class Camera2D {
	vec2 wCenter; 
	vec2 wSize;  
public:
	Camera2D() : wCenter(0.0f, 0.0f), wSize(20.0f, 20.0f) { }
	void setCentre(vec2 newcentre) {
		wCenter = newcentre;
	}
	mat4 V() { return TranslateMatrix(-wCenter); }
	mat4 P() { return ScaleMatrix(vec2(2.0f / wSize.x, 2.0f / wSize.y)); }

	mat4 Vinv() { return TranslateMatrix(wCenter); }
	mat4 Pinv() { return ScaleMatrix(vec2(wSize.x / 2.0f, wSize.y / 2.0f)); }

	void Zoom(float s) { wSize = wSize * s; }
	void Pan(vec2 t) { wCenter = wCenter + t; }
};

Camera2D camera;		

int compareVec4ByX(const void* v1, const void* v2){
	const float x1 = (float)((vec4*)v1)->x;
	const float x2 = (float)((vec4*)v2)->x;
	return x2 < x1 ? 1 : x2 == x1 ? 0 : -1;
}



class KochanekBartelsSpline {
	unsigned int vaoCurve, vboCurve;
	float tension, bias, continuity;
	std::vector<vec4> wCtrlPoints;
public:
	KochanekBartelsSpline(float t, float c, float b );

	void addSomeControlPoints(float y) {
		AddControlPointByCord(-119.0f, y);
		AddControlPointByCord(-117.0f, y);
		AddControlPointByCord(-17.0f, y);
		AddControlPointByCord(-15.0f, y);
		AddControlPointByCord(15.0f, y);
		AddControlPointByCord(17.0f, y);
		AddControlPointByCord(117.0f, y);
		AddControlPointByCord(119.0f, y);
	}

	void AddControlPoint(float cX, float cY) {
		vec4 wVertex = vec4(cX, cY, 0.0f, 1.0f) * camera.Pinv() * camera.Vinv();
		wCtrlPoints.push_back(wVertex);
		qsort(&wCtrlPoints[0], wCtrlPoints.size(), sizeof(vec4), compareVec4ByX);
	}

	void AddControlPointByCord(float cX, float cY) {
		vec4 wVertex = vec4(cX, cY, 0.0f, 1.0f);
		wCtrlPoints.push_back(wVertex);
		qsort(&wCtrlPoints[0], wCtrlPoints.size(), sizeof(vec4), compareVec4ByX);
	}


	void Draw() {
		int location = glGetUniformLocation(gpuProgram.getId(), "color");
		glUniform4f(location, 33 / 256.0f, 161 / 256.0f, 30 / 256.0f, 1.0f); 

		if (wCtrlPoints.size() >= 2) {
			std::vector<float> vertexData;
			for (int x = -150; x < 550; x++) {
				const float i = ((float)x / 20.0f) - 10.0f;
				vec4 wVertex(i, calculateY(i), 0, 1);
				vertexData.push_back(wVertex.x);
				vertexData.push_back(wVertex.y);
				vertexData.push_back(wVertex.x);
				vertexData.push_back(-20.0f);
			}

			glBindVertexArray(vaoCurve);
			glBindBuffer(GL_ARRAY_BUFFER, vboCurve);
			glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), &vertexData[0], GL_DYNAMIC_DRAW);
			glDrawArrays(GL_TRIANGLE_STRIP, 0, 1400);

		}
	}


	float calculateY(const float x) {

		vec4* leftside = &wCtrlPoints[0];
		vec4* actual = &wCtrlPoints[1];
		vec4* rightside = nullptr;
		vec4* toTheToRight = nullptr;
		for (unsigned int i = 2; i < wCtrlPoints.size() - 2; i++) {
			if (wCtrlPoints[i].x > x) {
				leftside = &wCtrlPoints[i - 2];
				actual = &wCtrlPoints[i - 1];
				rightside = &wCtrlPoints[i];
				toTheToRight = &wCtrlPoints[i + 1];
				break;
			}
		}
		if (rightside == nullptr)
			return 0;
		const vec4 v0 = ((*rightside - *actual) / (rightside->x - actual->x) +
						(*actual - *leftside) / (actual->x - leftside->x))  *  (1 - tension ) / 2.0f;
		
		const vec4 v1 = ((*toTheToRight - *rightside) / (toTheToRight->x - rightside->x) + 
						(*rightside - *actual) / (rightside->x - actual->x))  * (1 - tension) / 2.0f;
		const float Dt = rightside->x - actual->x;
		const float dt = x - actual->x;
		const vec4 a3 = ((*actual - *rightside) * 2 / powf(Dt, 3) + (v1 + v0) / powf(Dt, 2));
		const vec4 a2 = ((*rightside - *actual) * 3 / powf(Dt, 2) - (v1 + v0 * 2) / Dt);
		const vec4 a1 = v0;
		const vec4 a0 = *actual;
		const float res = ( a3* powf(dt, 3) + a2 * powf(dt,2) + a1 * dt + a0).y;
		return res;
	}


	float CalculateDeriative(const float x) {
		vec4* leftside = &wCtrlPoints[0];
		vec4* actual = &wCtrlPoints[1];
		vec4* rightside = nullptr;
		vec4* toTheToRight = nullptr;
		for (unsigned int i = 2; i < wCtrlPoints.size() - 2; i++) {
			if (wCtrlPoints[i].x > x) {
				leftside = &wCtrlPoints[i - 2];
				actual = &wCtrlPoints[i - 1];
				rightside = &wCtrlPoints[i];
				toTheToRight = &wCtrlPoints[i + 1];
				break;
			}
		}
		const vec4 v0 = ((*rightside - *actual) / (rightside->x - actual->x) +
			(*actual - *leftside) / (actual->x - leftside->x))  *  (1 - tension) / 2.0f;

		const vec4 v1 = ((*toTheToRight - *rightside) / (toTheToRight->x - rightside->x) +
						(*rightside - *actual) / (rightside->x - actual->x))  * (1 - tension) / 2.0f;

		const float Dt = rightside->x - actual->x;
		const float dt = x - actual->x;
		const vec4 a3 = ((*actual - *rightside) * 2 / powf(Dt, 3) + (v1 + v0) / powf(Dt, 2));
		const vec4 a2 = ((*rightside - *actual) * 3 / powf(Dt, 2) - (v1 + v0 * 2) / Dt);
		const vec4 a1 = v0;
		const vec4 a0 = *actual;
		const float res = ( a3 * 3 * powf(dt, 2) + a2* 2 * dt + a1).y;
		return res;
	}
};
KochanekBartelsSpline::KochanekBartelsSpline(float t, float c, float b) :
	tension(t), continuity(c), bias(b) {
	glGenVertexArrays(1, &vaoCurve);
	glBindVertexArray(vaoCurve);
	glGenBuffers(1, &vboCurve); 
	glBindBuffer(GL_ARRAY_BUFFER, vboCurve);
	glEnableVertexAttribArray(0);  
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), NULL); 

	addSomeControlPoints(-1.2f);
}

KochanekBartelsSpline* kb;

void addPointToBuffer(float x, float y, std::vector<float> &tarolo, vec3 rgb, GLuint vbo) {
	tarolo.push_back(x);
	tarolo.push_back(y);
	tarolo.push_back(rgb.x);
	tarolo.push_back(rgb.y);
	tarolo.push_back(rgb.z);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, tarolo.size() * sizeof(float), &tarolo[0], GL_DYNAMIC_DRAW);
}

class Biker {
	GLuint vao, kullovao, vbo, kullovbo, vaobody, vbobody;	
	std::vector<float>  vertexData, kullodata, bodydata; 
	vec2 wTranslate,shoulderCenter,center, drawingcentre;
	float radius, rotate = 0.0f, timePassedSinceStart = 0.0f;
	const float F = 18.0f, m = 1.0f, g = 9.8f, rho = 0.8f, thighBoneLength = 1.6f, shinBoneLength = 1.7f;
	Orientation orientation = right;
	KochanekBartelsSpline* kb;

	float CalculateRotation(vec2 asd) {
		float circumference = 2.0f * radius * PI;
		float motion = sqrtf(powf(asd.x, 2.0f) + powf(asd.y, 2.0f));
		return -2.0f * PI*motion / circumference;
	}
public:
	Biker() {}
	
	void setSpline(KochanekBartelsSpline* kb) {
		this->kb = kb;
	}
	void moveCenterByVec(vec2 asd) {
		fixOrientation(asd);
		fixCentreY();
		center = center + asd;
		float derivative = orientation * kb->CalculateDeriative(center.x);
		float alpha = atanf(derivative);
		vec2 N = vec2(-sinf(alpha), cosf(alpha));
		float denim = sqrtf(N.x*N.x + N.y * N.y);

		N = N * (1.0f / denim);
		if (N.y < 0.0f && (center + (N * radius)).y < kb->calculateY((center + (N * radius)).x)) {
			N = N * (-1.0f);
		}
		if (orientation != right && N.x < 0.0f) {
			N.x = N.x * (-1.0f);
		}
		if ((orientation == left && derivative < 0.0f && N.x > 0.0f)) {
			N.x = N.x * (-1.0f);
		}

		drawingcentre = center + (N * radius);
		rotate += orientation * CalculateRotation(asd);
	}

	void Create() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo); 
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glEnableVertexAttribArray(0);  
		glEnableVertexAttribArray(1);  
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(0)); 
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));

		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), &vertexData[0], GL_DYNAMIC_DRAW);

		{
			glGenVertexArrays(1, &kullovao);
			glBindVertexArray(kullovao);
			glGenBuffers(1, &kullovbo); 
			glBindBuffer(GL_ARRAY_BUFFER, kullovbo);
			glEnableVertexAttribArray(0); 
			glEnableVertexAttribArray(1);  
			glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(0)); 
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));
			glBindBuffer(GL_ARRAY_BUFFER, kullovbo);
			glBufferData(GL_ARRAY_BUFFER, kullodata.size() * sizeof(float), &kullodata[0], GL_DYNAMIC_DRAW);
		}
		{
			glGenVertexArrays(1, &vaobody);
			glBindVertexArray(vaobody);
			glGenBuffers(1, &vbobody); 
			glBindBuffer(GL_ARRAY_BUFFER, vbobody);
			glEnableVertexAttribArray(0); 
			glEnableVertexAttribArray(1);  
			glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(0)); 
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));

			glBindBuffer(GL_ARRAY_BUFFER, vbo);
			glBufferData(GL_ARRAY_BUFFER, bodydata.size() * sizeof(float), &bodydata[0], GL_DYNAMIC_DRAW);
		}

		radius = 0.8f;
		center = vec2(2.4f, -1.2f);
		drawingcentre = vec2(2.4f, center.y + radius);
	}


	void moveRight() {
		float derivative = orientation * kb->CalculateDeriative(center.x);
		float alpha = atanf(derivative);
		vec2 V = vec2(cosf(alpha), sinf(alpha));
		float denim = sqrtf(V.x*V.x + V.y * V.y);
		V = V * (1.0f / (300.0f * denim));
		float v = (F - m * g*sin(alpha)) / rho;
		moveCenterByVec(V*v);
	}

	void moveLeft() {
		orientation = left;
		float derivative = orientation * kb->CalculateDeriative(center.x);
		float alpha = atanf(derivative);
		vec2 V = vec2(-cosf(alpha), sinf(alpha));
		float denim = sqrtf(V.x*V.x + V.y * V.y);
		V = V * (1.0f / (300.0f * denim));
		float v = (F - m * g*sin(alpha)) / rho;
		moveCenterByVec(V*v);
	}

	void fixCentreY() {
		center.y = kb->calculateY(center.x);
		float derivative = orientation * kb->CalculateDeriative(center.x);
		float alpha = atanf(derivative);
		vec2 N = vec2(-sinf(alpha), cosf(alpha));
		float denim = sqrtf(N.x*N.x + N.y * N.y);

		N = N * (1.0f / denim);
		if (N.y < 0.0f) {
			N = N * (-1.0f);
		}
		drawingcentre = center + (N * radius);
	}

	void fixOrientation(vec2 asd) {
		Orientation tempOri = asd.x > 0 ? right : left;
		if (tempOri != orientation)
			fixCentreY();
		orientation = tempOri;
	}
	mat4 M() {
		return mat4(1.0f, 0.0f, 0.0f, 0.0f,
					0.0f, 1.0f, 0.0f, 0.0f,
					0.0f, 0.0f, 1.0f, 0.0f,
					wTranslate.x, wTranslate.y, 0.0f, 1.0f); 
	}
	mat4 Mkullok() {
		return mat4(1.0f, 0.0f, 0.0f, 0.0f,
					0.0f, 1.0f, 0.0f, 0.0f,
					0.0f, 0.0f, 1.0f, 0.0f,
					0.0f, 0.0f, 0.0f, 1.0f); 
	}

	void AddPointByCord(float x, float y) {
		vertexData.push_back(x);
		vertexData.push_back(y);
		vertexData.push_back(0.0f); 
		vertexData.push_back(0.0f); 
		vertexData.push_back(0.0f); 
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), &vertexData[0], GL_DYNAMIC_DRAW);
	}

	void makeCircle() {
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBindVertexArray(vao);
		for (float i = 0.0f; i <= 2.0f * PI; i += 2.0f * PI / 100.0f) {
			AddPointByCord(cos(i)*radius + drawingcentre.x, sin(i)*radius + drawingcentre.y);
		}
	}

	void makekullok() {
		glBindVertexArray(kullovao);
		for (float i = 0.0f; i <= 2 * PI; i += 2.0f * PI / 6.0f) {
			addPointToBuffer(cos(i + rotate)*radius + drawingcentre.x, sin(i + rotate)*radius + drawingcentre.y, kullodata, vec3(0.0f, 0.0f, 0.0f), kullovbo);
			addPointToBuffer(drawingcentre.x, drawingcentre.y, kullodata, vec3(0.0f, 0.0f, 0.0f), kullovbo);
		}
	}
	void makeBody() {
		glBindVertexArray(vaobody);
		makeLeg(left);
		makeLeg(right);
		makeSeat();
		glLineWidth(1.5f);
		makeUpperBody();
		makeArms();
		makeHead();
	}

	void makeHead() {
		vec2 headcentre = vec2(shoulderCenter.x, shoulderCenter.y + 0.4f);
		for (float i = -PI/2.0f; i <= 1.5f*PI + 0.001f; i += PI / 14.0f) {
			addPointToBuffer(headcentre.x + 0.4f* cosf(i), headcentre.y + 0.4f*sinf(i), bodydata, vec3(0.0f, 0.0f, 0.0f), vbobody);
		}
	}
	void makeArms() {
		addPointToBuffer(shoulderCenter.x+1.2f, shoulderCenter.y - 2.0f, bodydata, vec3(0.0f, 0.0f, 0.0f), vbobody);
		addPointToBuffer(shoulderCenter.x, shoulderCenter.y, bodydata, vec3(0.0f, 0.0f, 0.0f), vbobody);
		addPointToBuffer(shoulderCenter.x - 1.2f, shoulderCenter.y - 2.0f, bodydata, vec3(0.0f, 0.0f, 0.0f), vbobody);
		addPointToBuffer(shoulderCenter.x, shoulderCenter.y, bodydata, vec3(0.0f, 0.0f, 0.0f), vbobody);
	}

	void makeUpperBody() {
		shoulderCenter.x = drawingcentre.x + 0.9f * (cosf(timePassedSinceStart)) / 3;
		const float c = 2.18f;
		const float b = fabsf(drawingcentre.x - shoulderCenter.x);
		const float a = sqrtf(powf(c, 2) - powf(b, 2));
		shoulderCenter.y = drawingcentre.y + 2.18f + a;
		addPointToBuffer(shoulderCenter.x, shoulderCenter.y, bodydata, vec3(0, 0, 0), vbobody);
	}

	void makeSeat() {
		addPointToBuffer(drawingcentre.x + 0.3f, drawingcentre.y + 2.18f, bodydata, vec3(0.0f, 0.0f, 0.0f), vbobody);
		addPointToBuffer(drawingcentre.x - 0.3f, drawingcentre.y + 2.18f, bodydata, vec3(0.0f, 0.0f, 0.0f), vbobody);
		addPointToBuffer(drawingcentre.x, drawingcentre.y + 2.18f, bodydata, vec3(0.0f, 0.0f, 0.0f), vbobody);
	}

	
	
	void makeLeg(const Orientation leg) {
		const float legf = leg == right ? 1.0f : -1.0f;
		const float offset = PI / 2.0f * legf;
		const vec2 P1 = vec2(drawingcentre.x, drawingcentre.y + 2.2f);
		const vec2 P0 = vec2(cos(rotate + offset)*radius*0.75f + drawingcentre.x, sin(rotate + offset)*radius*0.75f + drawingcentre.y);
		const float d = sqrtf(powf(P1.x - P0.x, 2.0f) + powf(P1.y - P0.y, 2.0f));
		const float r0 = shinBoneLength;
		const float r1 = thighBoneLength;
		const float a = (r0*r0 - r1 * r1 + d * d) / (2 * d);
		const float h = sqrtf(r0*r0 - a * a);
		
		const vec2 P2 = P0 + ((P1 - P0)*a) *(1.0f/ d);

		const vec2 V = (P1 - P2)* (1 / sqrtf((P1 - P2).x*(P1 - P2).x + (P1 - P2).y * (P1 - P2).y));
		const vec2 N = vec2(-V.y, V.x);

		const vec2 asd = orientation == left ? P2 + N * h : P2 - N * h;
		
		addPointToBuffer(P1.x, P1.y, bodydata, vec3(0.0f, 0.0f, 0.0f), vbobody);
		addPointToBuffer(asd.x, asd.y, bodydata, vec3(0.0f, 0.0f, 0.0f), vbobody);
		addPointToBuffer(P0.x, P0.y, bodydata, vec3(0.0f, 0.0f, 0.0f), vbobody);
		addPointToBuffer(asd.x, asd.y, bodydata, vec3(0.0f, 0.0f, 0.0f), vbobody);
		addPointToBuffer(P1.x, P1.y, bodydata, vec3(0.0f, 0.0f, 0.0f), vbobody);
	}

	void Draw() {
		glLineWidth(3.0f);
		int location = glGetUniformLocation(gpuProgram.getId(), "color");
		clearPreviousData();
		glUniform4f(location, 0.0f, 0.0f, 0.0f, 1.0f);
		makeCircle();
		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glDrawArrays(GL_LINE_LOOP, 0, vertexData.size() / 5);


		makekullok();
		glBindVertexArray(kullovao);
		glBindBuffer(GL_ARRAY_BUFFER, kullovbo);
		glBufferData(GL_ARRAY_BUFFER, kullodata.size() * sizeof(float), &kullodata[0], GL_DYNAMIC_DRAW);
		glDrawArrays(GL_LINE_STRIP, 0, kullodata.size() / 5);


		makeBody();
		glBindVertexArray(vaobody);
		glBindBuffer(GL_ARRAY_BUFFER, vbobody);
		glDrawArrays(GL_LINE_STRIP, 0, bodydata.size() / 5);
		glLineWidth(2.0f);
	}

	vec2 getCentre() {
		return center;
	}
	float timePassedSinceLastFrame = 0.0f;
	void animate(float et) {
		timePassedSinceStart += et;
		timePassedSinceLastFrame += et;
		if (timePassedSinceLastFrame > 0.02f) {  // 50 FPS max
			timePassedSinceLastFrame = 0.0f;
			if (isAtEdge() && !justChangedOrientation) {
				changeOrientation();
				justChangedOrientation = true;
			}
			Move();
			if (movesSinceLastChange > 50) {
				justChangedOrientation = false;
				movesSinceLastChange = 0;
			}
			if (justChangedOrientation)
				movesSinceLastChange++;
		}
	}
private:
	void clearPreviousData() {
		vertexData.clear();
		kullodata.clear();
		bodydata.clear();
	}

	bool justChangedOrientation = false;
	int movesSinceLastChange = 0;
	
	bool isAtEdge() {
		return center.x < -10.0f || center.x > 10.0f;
	}

	void changeOrientation() {
		orientation = orientation == right ? left : right;
	}
	void Move() {
		orientation == right ? moveRight() : moveLeft();
	}

};

Biker biker;

class TexturedQuad {
	unsigned int vao, vbo[2];
	vec2 vertices[4], uvs[4];
public:
	TexturedQuad() {
		vertices[0] = vec2(-20.0f, -20.0f); uvs[0] = vec2(0.0f, 0.0f);
		vertices[1] = vec2(20.0f, -20.0f);  uvs[1] = vec2(1.0f, 0.0f);
		vertices[2] = vec2(20.0f, 20.0f);   uvs[2] = vec2(1.0f, 1.0f);
		vertices[3] = vec2(-20.0f, 20.0f);  uvs[3] = vec2(0.0f, 1.0f);
	}
	void Create() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);		
		glGenBuffers(2, vbo);	
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); 
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);	  
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);    

		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); 
		glBufferData(GL_ARRAY_BUFFER, sizeof(uvs), uvs, GL_STATIC_DRAW);	  
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);     
	}

	mat4 BikeFollowingMatrice() {
		return mat4(1.0f*appliedZoom,				 0.0f,							0.0f, 0.0f,
					0.0f,							 1.0f*appliedZoom,				0.0f, 0.0f,
					0.0f,					         0.0f,							1.0f, 0.0f,
					biker.getCentre().x / 10.0f,     biker.getCentre().y / 10.0f,   0.0f, 1.0f); 
	}
	void Draw() {
		glBindVertexArray(vao);	

		mat4 MVPTransform = camera.V() * camera.P();
		if (FollowingBike)
			MVPTransform = MVPTransform * BikeFollowingMatrice();
		MVPTransform.SetUniform(backGroundMaker.getId(), "MVP");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	
	}
};

TexturedQuad quad;


void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	backGroundMaker.Create(vertexSourceForBackground, fragmentSourceForBackground, "outColor");
	quad.Create();
	gpuProgram.Create(vertexSource, fragmentSource, "outColor");
	gpuProgram.Use();
	glLineWidth(2.0f); 
	kb = new KochanekBartelsSpline(-1.0f, 0.0f, 0.0f);
	biker.Create();
	biker.setSpline(kb);


	glGenVertexArrays(1, &vao);	
	glBindVertexArray(vao);		

	unsigned int vbo;		
	glGenBuffers(1, &vbo);	
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	glEnableVertexAttribArray(0);  
	glVertexAttribPointer(0,2, GL_FLOAT, GL_FALSE, 0, NULL); 		     
	glutPostRedisplay();
}


void onDisplay() {
	glClearColor(159.0f / 256.0f, 207.0f / 256.0f, 230.0f / 256.0f, 0.98f);
	glClear(GL_COLOR_BUFFER_BIT); 

	int location = glGetUniformLocation(gpuProgram.getId(), "color");
	glUniform4f(location, 1.0f, 1.0f, 0.0f, 1.0f); 

	float MVPtransf[4][4] = { 1.0f, 0.0f, 0.0f, 0.0f,    
							  0.0f, 1.0f, 0.0f, 0.0f,    
							  0.0f, 0.0f, 1.0f, 0.0f,
							  0.0f, 0.0f, 0.0f, 1.0f };

	glBindVertexArray(vao);  
	
	if (FollowingBike)
		camera.setCentre(biker.getCentre());

	backGroundMaker.Use();
	quad.Draw();
	gpuProgram.Use();
	mat4 VPTransform = camera.V() * camera.P();
	VPTransform.SetUniform(gpuProgram.getId(), "MVP");
	kb->Draw();
	biker.Draw();
	glutSwapBuffers(); 

}
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') {
		biker.moveRight();
		glutPostRedisplay();
	}
	if (key == 'a') {
		biker.moveLeft();
		glutPostRedisplay();
	}
	if (key == ' ') {
		if (!FollowingBike) {
			camera.Zoom(appliedZoom);
		}
		FollowingBike = !FollowingBike; {
			if (!FollowingBike) {
				camera.setCentre(vec2(0.0f, 0.0f));
				camera.Zoom(1.0f /appliedZoom);

			}
		}

	}
}

void onKeyboardUp(unsigned char key, int pX, int pY) {
}

void onMouseMotion(int pX, int pY) {
}

void onMouse(int button, int state, int pX, int pY) { 
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) { 
		float cX = 2.0f * pX / windowWidth - 1;	
		float cY = 1.0f - 2.0f * pY / windowHeight;
		kb->AddControlPoint(cX, cY);
		glutPostRedisplay();
	}
}
const float dt = 0.01f;
float lastFrameTime = 0.0f;
bool hadWaited = false;

void onIdle() {
	if (!hadWaited) {
		hadWaited = true;
	}
	static float tend = 0.0f;
	float tstart = tend;
	tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
	int lefutott = 0;
	float elteltido = 0.0f;
	for (float t = tstart; t < tend; t += dt) {
		float Dt = fmin(dt, tend - t);
		biker.animate(Dt);
		lefutott++;
		elteltido += Dt;
	}
	glutPostRedisplay();
}
