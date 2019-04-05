//=============================================================================================
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Koncz Adam
// Neptun : MOENI1
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"
// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char * const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0

	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
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
	
	uniform float tension = 0.85f;
	 vec2 leftside;
	 vec2 actual;
	 vec2 rightside;
	 vec2 twoToTheRight;
	    vec2 controlPoints[8]=vec2[8](
		vec2(-0.1, 0.6),
		vec2(0.2, 0.7),
		vec2(0.4,0.45),
		vec2(0.6,0.7),
		vec2(0.8,0.65),
		vec2(1,0.4),
		vec2(1.2,0.6),
		vec2(1.4, 0.6)
		
);	

	in vec2 texCoord;			// variable input: interpolated texture coordinates
	out vec4 outColor;		// output that goes to the raster memory as told by glBindFragDataLocation

float H0(float s) {
		return 2*s*s*s - 3*s*s + 1;
}
float H1(float s) {
	return -2 * s*s*s + 3 * s*s ;
}
float H2(float s) {
	return  s*s*s - 2 * s*s + s;
}
float H3(float s) {
	return s*s*s - s*s;
}
bool isHill(vec2 Coord){
		float y = 0;
		for(int i = 1; i < 7; i++){
			if(controlPoints[i].x > Coord.x){
				leftside = controlPoints[i-2];
				actual = controlPoints[i-1];
				rightside = controlPoints[i];
				twoToTheRight = controlPoints[i+1];
				break;
			}
		}
		float deltaI = rightside.x - actual.x;
		float TiI = ((1 - tension) / 2)*(twoToTheRight.y - rightside.y) + ((1 - tension)/ 2)*(rightside.y - actual.y);
		float TiO = ((1 - tension)/ 2)*(rightside.y - actual.y) + ((1 - tension)/ 2)*(actual.y - leftside.y);
		y = H0((Coord.x - actual.x) / deltaI) * actual.y;
		y += H1((Coord.x - actual.x) / deltaI) * rightside.y;
		y += H2((Coord.x - actual.x) / deltaI) * TiO;
		y += H3((Coord.x - actual.x) / deltaI) * TiI ;
		if(y > Coord.y)
			return true;
		return false;
	}

float plot (vec2 st, float pct){
  return  smoothstep( pct-0.01, pct, st.y) -
          smoothstep( pct, pct+0.01, st.y);
}
	void main() {
		if (isHill(texCoord)) {
			outColor = vec4(131/255.0f, 204/255.0f, 211/255.0f, 1); 
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
GPUProgram gpuProgram; // vertex and fragment shaders
GPUProgram backGroundMaker;
unsigned int vao;	   // virtual world on the GPU

class Camera2D {
	vec2 wCenter; // center in world coordinates
	vec2 wSize;   // width and height in world coordinates
public:
	Camera2D() : wCenter(0, 0), wSize(20, 20) { }
	void setCentre(vec2 newcentre) {
		wCenter = newcentre;
	}
	mat4 V() { return TranslateMatrix(-wCenter); }
	mat4 P() { return ScaleMatrix(vec2(2 / wSize.x, 2 / wSize.y)); }

	mat4 Vinv() { return TranslateMatrix(wCenter); }
	mat4 Pinv() { return ScaleMatrix(vec2(wSize.x / 2, wSize.y / 2)); }

	void Zoom(float s) { wSize = wSize * s; }
	void Pan(vec2 t) { wCenter = wCenter + t; }
};

Camera2D camera;		// 2D camera

int compareVec4ByX(const void* v1, const void* v2) {
	float x1 = ((vec4*)v1)->x;
	float x2 = ((vec4*)v2)->x;
	if (x2 < x1)
		return 1;
	if (x2 == x1)
		return 0;
	return -1;
}

class KochanekBartelsSpline {
	unsigned int vaoCurve, vboCurve;
	unsigned int vaoCtrlPoints, vboCtrlPoints;
	std::vector<float> ts;  // knots
	float tension, bias, continuity;
	float L(int i, float t) {
		float Li = 1.0f;
		for (unsigned int j = 0; j < wCtrlPoints.size(); j++)
			if (j != i) Li *= (t - ts[j]) / (ts[i] - ts[j]);
		return Li;
	}
	std::vector<vec4> wCtrlPoints;		// coordinates of control points
	float H0(float);
	float H1(float);
	float H2(float);
	float H3(float);
public:
	KochanekBartelsSpline(float t = 0.0f, float c = 0.0f, float b = 0.0f);

	void addSomeControlPoints(float y) {
		AddControlPointByCord(-19.0f, y);
		AddControlPointByCord(-17.0f, y);
		AddControlPointByCord(-15.0f, y);
		AddControlPointByCord(15.0f, y);
		AddControlPointByCord(17.0f, y);
		AddControlPointByCord(19.0f, y);
	}

	void AddControlPoint(float cX, float cY) {
		ts.push_back((float)wCtrlPoints.size());
		vec4 wVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
		wCtrlPoints.push_back(wVertex);
		qsort(&wCtrlPoints[0], wCtrlPoints.size(), sizeof(vec4), compareVec4ByX);
	}

	void AddControlPointByCord(float cX, float cY) {
		ts.push_back((float)wCtrlPoints.size());
		vec4 wVertex = vec4(cX, cY, 0, 1);
		wCtrlPoints.push_back(wVertex);
		qsort(&wCtrlPoints[0], wCtrlPoints.size(), sizeof(vec4), compareVec4ByX);
	}


	void Draw() {
		mat4 VPTransform = camera.V() * camera.P();
		gpuProgram.Use();
		VPTransform.SetUniform(gpuProgram.getId(), "MVP");

		int colorLocation = glGetUniformLocation(gpuProgram.getId(), "color");

		if (wCtrlPoints.size() > 0) {	// draw control points
			glBindVertexArray(vaoCtrlPoints);
			glBindBuffer(GL_ARRAY_BUFFER, vboCtrlPoints);
			glBufferData(GL_ARRAY_BUFFER, wCtrlPoints.size() * 4 * sizeof(float), &wCtrlPoints[0], GL_DYNAMIC_DRAW);
			if (colorLocation >= 0) glUniform3f(colorLocation, 1, 0, 0);
			glPointSize(10.0f);
			int location = glGetUniformLocation(gpuProgram.getId(), "color");
			glUniform4f(location, 1.0f, 0.0f, 0.0f, 0.0f); // 3 floats*/
			glDrawArrays(GL_POINTS, 0, wCtrlPoints.size());
		}


		int location = glGetUniformLocation(gpuProgram.getId(), "color");
		glUniform4f(location, 33 / 256.0f, 161 / 256.0f, 30 / 256.0f, 1.0f); // 3 floats

		if (wCtrlPoints.size() >= 2) {	// draw curve
			std::vector<float> vertexData;
			for (int x = 0; x < 400; x++) {
				const float i = ((float)x / 20.0f) - 10.0f;
				vec4 wVertex(i, calculateY(i), 0, 1);
				vertexData.push_back(wVertex.x);
				vertexData.push_back(wVertex.y);
				vertexData.push_back(wVertex.x);
				vertexData.push_back(-10.0f);
			}
			// copy data to the GPU
			glBindVertexArray(vaoCurve);
			glBindBuffer(GL_ARRAY_BUFFER, vboCurve);
			glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), &vertexData[0], GL_DYNAMIC_DRAW);
			if (colorLocation >= 0)
				glUniform3f(colorLocation, 1, 1, 0);
			glDrawArrays(GL_TRIANGLE_STRIP, 0, 800);

		}

	}
	void sortControlpoints() {
		for (int i = wCtrlPoints.size() - 1; i > 0; i--) {
			for (int j = 0; j < i - 1; j++) {
				if (wCtrlPoints[j].x > wCtrlPoints[j + 1].x) {
					vec4 Temp = wCtrlPoints[j];
					wCtrlPoints[j] = wCtrlPoints[j + 1];
					wCtrlPoints[j + 1] = Temp;
				}
			}
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

		if (rightside == nullptr) {
			return -6.2f;
		}

		float deltaI = rightside->x - actual->x;
		//Choosing Tangent Vectors
		//incoming target vector
		//float TiI = ((1 - tension)*(1 + continuity)*(1 - bias) / 2)*(rightside->y - actual->y) + ((1 - tension)*(1 - continuity)*(1 + bias) / 2)*(actual->y - leftside->y);
		float TiI = ((1 - tension)*(1 + continuity)*(1 - bias) / 2)*(toTheToRight->y - rightside->y) + ((1 - tension)*(1 - continuity)*(1 + bias) / 2)*(rightside->y - actual->y);

		//outgoing target vec
		float TiO = ((1 - tension)*(1 - continuity)*(1 - bias) / 2)*(rightside->y - actual->y) + ((1 - tension)*(1 + continuity)*(1 + bias) / 2)*(actual->y - leftside->y);

		float res = 0;
		res += H0((x - actual->x) / deltaI)  *  actual->y;
		res += H1((x - actual->x) / deltaI)  *  rightside->y;
		res += H2((x - actual->x) / deltaI)  *  TiO;
		res += H3((x - actual->x) / deltaI)  *  TiI;
		return res;

	}
	float calculateYnew(float x) {
		vec4* leftside = nullptr;
		vec4* actual = &wCtrlPoints[1];
		vec4* rightside = nullptr;
		vec4* toTheToRight = nullptr;
		for (unsigned int i = 1; i < wCtrlPoints.size() - 1; i++) {
			if (wCtrlPoints[i].x > x) {
				if (i > 1)
					leftside = &wCtrlPoints[i - 2];
				actual = &wCtrlPoints[i - 1];
				rightside = &wCtrlPoints[i + 0];
				if (i != wCtrlPoints.size() - 1)
					toTheToRight = &wCtrlPoints[i + 1];
				break;
			}
		}

		if (rightside == nullptr) {

			return -6.2f;
		}

		float deltaI = rightside->x - actual->x;
		//incoming target vector
		//float TiI = ((1 - tension)*(1 + continuity)*(1 - bias) / 2)*(rightside->y - actual->y) + ((1 - tension)*(1 - continuity)*(1 + bias) / 2)*(actual->y - leftside->y);
		float TiI = ((1 - tension)*(1 + continuity)*(1 - bias) / 2)*(toTheToRight->y - rightside->y) + ((1 - tension)*(1 - continuity)*(1 + bias) / 2)*(rightside->y - actual->y);

		//outgoing target vec
		float TiO = ((1 - tension)*(1 - continuity)*(1 - bias) / 2)*(rightside->y - actual->y) + ((1 - tension)*(1 + continuity)*(1 + bias) / 2)*(actual->y - leftside->y);

		float res = 0;
		//res += H0((x - actual->x) / deltaI)  *  actual->y;
		//res += H1((x - actual->x) / deltaI)  *  rightside->y;
		//res += H2((x - actual->x) / deltaI) * 1 * TiO;
		//res += H3((x - actual->x) / deltaI) * 1 * TiI;

		float v0;
		v0 = leftside == nullptr ? 1 : (actual->y - leftside->y) / (actual->x - leftside->x)*
			(1 - tension)*(1 + continuity)*(1 + bias) / 2 +
			(rightside->y - actual->y) / (rightside->x - actual->x)*
			(1 - tension)*(1 - continuity)*(1 - bias) / 2;
		float v1;
		v1 = toTheToRight == nullptr ? 1 : (rightside->y - actual->y) / (rightside->x - actual->x)*
			(1 - tension)*(1 - continuity)*(1 + bias) / 2 +
			(toTheToRight->y - rightside->y) / (toTheToRight->x / rightside->x)*
			(1 - tension)*(1 + continuity)*(1 + bias) / 2;
		
		//v0 = TiO;
		//v1 = TiI;
		float Dt = rightside->x - actual->x;
		float f0 = actual->y;
		float f1 = rightside->y;
		float T = (x - actual->x)/Dt;
		float a = (v1 + v0) / Dt / Dt - (f0 - f1) * 2 / Dt / Dt / Dt;
		float b = (f1 - f0) * 3 / Dt / Dt - (v1 + v0 * 2) / Dt;
		float c = v0;
		float d = f0;

		return (a*T*T*T + b * T*T + c * T + d);
	}


	float CalculateDeriative(float x) {
		float y = calculateY(x);
		float yp1 = calculateY(x + 0.01f);
		return -(yp1 - y) / (x - (x + 0.01f));
	}

	float CalcDif2(float x) {
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
		float deltaI = rightside->x - actual->x;
		float TiI = ((1 - tension)*(1 + continuity)*(1 - bias) / 2)*(toTheToRight->y - rightside->y) + ((1 - tension)*(1 - continuity)*(1 + bias) / 2)*(rightside->y - actual->y);
		float TiO = ((1 - tension)*(1 - continuity)*(1 - bias) / 2)*(rightside->y - actual->y) + ((1 - tension)*(1 + continuity)*(1 + bias) / 2)*(actual->y - leftside->y);
		float res = 0;
		float s = ((x - actual->x) / deltaI);
		/*res += (6 * s*s - 6 * s) * (x / deltaI) * actual->y;
		res += (-6 * s*s + 6 * s) * (x / deltaI) *  rightside->y;
		res += (3 * s*s - 4 * s + 1) * (x / deltaI) *deltaI* TiO;
		res += (3 * s*s - 2 * s) * (x / deltaI) *deltaI* TiI;*/
		const float param = (x - actual->x) / deltaI;
		const float tti = x / deltaI;
		res += (6 * param*param - 6 * param)* tti*actual->y;
		res += (-6 * param*param + 6 * param) *tti* rightside->y;
		res += (3 * param*param - 4 * param + 1)* TiO * x;
		res += (3 * param*param - 2 * param)*TiI* x;
		return res;
	}
};
KochanekBartelsSpline::KochanekBartelsSpline(float t, float c, float b) :
	tension(t), continuity(c), bias(b) {
	glGenVertexArrays(1, &vaoCurve);
	glBindVertexArray(vaoCurve);

	glGenBuffers(1, &vboCurve); // Generate 1 vertex buffer object
	glBindBuffer(GL_ARRAY_BUFFER, vboCurve);
	// Enable the vertex attribute arrays
	glEnableVertexAttribArray(0);  // attribute array 0
	// Map attribute array 0 to the vertex data of the interleaved vbo
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), NULL); // attribute array, components/attribute, component type, normalize?, stride, offset

	// Control Points
	glGenVertexArrays(1, &vaoCtrlPoints);
	glBindVertexArray(vaoCtrlPoints);

	glGenBuffers(1, &vboCtrlPoints); // Generate 1 vertex buffer object
	glBindBuffer(GL_ARRAY_BUFFER, vboCtrlPoints);
	// Enable the vertex attribute arrays
	glEnableVertexAttribArray(0);  // attribute array 0
	// Map attribute array 0 to the vertex data of the interleaved vbo
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), NULL); // attribute array, components/attribute, component type, normalize?, stride, offset

	addSomeControlPoints(-3.2f);
}

float KochanekBartelsSpline::H0(float s) {
	return 2 * s*s*s - 3 * s*s + 1;
}
float KochanekBartelsSpline::H1(float s) {
	return -2 * s*s*s + 3 * s*s;
}
float KochanekBartelsSpline::H2(float s) {
	return  s * s*s - 2 * s*s + s;
}
float KochanekBartelsSpline::H3(float s) {
	return s * s*s - s * s;
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
	GLuint				vao, kullovao, vbo, kullovbo, vaobody, vbobody;	// vertex array object, vertex buffer object
	std::vector<float>  vertexData, kullodata, bodydata; // interleaved data of coordinates and colors
	vec2			    wTranslate;
	vec2 shoulderCenter;
	vec2 center, drawingcentre;
	float radius;
	vec4 offset = vec4(0, 0, 0, 0);
	float rotate = 0;
	Orientation orientation = right;
	KochanekBartelsSpline* kb;
	float timePassedSinceStart = 0;

	float CalculateRotation(vec2 asd) {
		float circumference = 2 * radius * PI;
		float motion = sqrtf(powf(asd.x, 2) + powf(asd.y, 2));
		return -2 * PI*motion / circumference;
	}
public:
	void setSpline(KochanekBartelsSpline* kb) {
		this->kb = kb;
	}
	void moveCenterByVec(vec2 asd) {
		/*
		orientation = asd.x > 0 ? right : left;
		center = center + asd;
		center.y = kb->calculateY(center.x)+0.69f;
		rotate += orientation*CalculateRotation(asd);*/

		fixOrientation(asd);
		fixCentreY();
		int ori = orientation;
		center = center + asd;
		float derivative = orientation * kb->CalculateDeriative(center.x);
		//printf("deriative: %f\n", derivative);
		float alpha = atanf(derivative);
		vec2 N = vec2(-sinf(alpha), cosf(alpha));
		float denim = sqrtf(N.x*N.x + N.y * N.y);

		//N.x = N.x / denim;
		//N.y = N.y / denim;
		N = N * (1 / denim);
		if (N.y < 0 && (center + (N * radius)).y < kb->calculateY((center + (N * radius)).x)) {
			N = N * (-1);
		}
		if (orientation != right && N.x < 0) {
			N.x = N.x * (-1);
		}
		if ((orientation == left && derivative < 0 && N.x > 0)) {
			N.x = N.x * (-1);
		}

		drawingcentre = center + (N * radius);

		//center.y = kb->calculateY(center.x) + 0.69f;
		//wTranslate = wTranslate + asd;
		rotate += orientation * CalculateRotation(asd);
	}



	void Create() {

		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0
		glEnableVertexAttribArray(1);  // attribute array 1
		// Map attribute array 0 to the vertex data of the interleaved vbo
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(0)); // attribute array, components/attribute, component type, normalize?, stride, offset
		// Map attribute array 1 to the color data of the interleaved vbo
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));

		// copy data to the GPU
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), &vertexData[0], GL_DYNAMIC_DRAW);


		{
			glGenVertexArrays(1, &kullovao);
			glBindVertexArray(kullovao);
			glGenBuffers(1, &kullovbo); // Generate 1 vertex buffer object
			glBindBuffer(GL_ARRAY_BUFFER, kullovbo);
			glEnableVertexAttribArray(0);  // attribute array 0
			glEnableVertexAttribArray(1);  // attribute array 1
			// Map attribute array 0 to the vertex data of the interleaved vbo
			glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(0)); // attribute array, components/attribute, component type, normalize?, stride, offset
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));
			// copy data to the GPU
			glBindBuffer(GL_ARRAY_BUFFER, kullovbo);
			glBufferData(GL_ARRAY_BUFFER, kullodata.size() * sizeof(float), &kullodata[0], GL_DYNAMIC_DRAW);
		}
		{
			glGenVertexArrays(1, &vaobody);
			glBindVertexArray(vaobody);
			glGenBuffers(1, &vbobody); // Generate 1 vertex buffer object
			glBindBuffer(GL_ARRAY_BUFFER, vbobody);
			// Enable the vertex attribute arrays
			glEnableVertexAttribArray(0);  // attribute array 0
			glEnableVertexAttribArray(1);  // attribute array 1
			// Map attribute array 0 to the vertex data of the interleaved vbo
			glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(0)); // attribute array, components/attribute, component type, normalize?, stride, offset
			// Map attribute array 1 to the color data of the interleaved vbo
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));

			// copy data to the GPU
			glBindBuffer(GL_ARRAY_BUFFER, vbo);
			glBufferData(GL_ARRAY_BUFFER, bodydata.size() * sizeof(float), &bodydata[0], GL_DYNAMIC_DRAW);
		}


		radius = 0.8f;
		drawingcentre = vec2(2.4f, -2.4f);
		center = vec2(2.4f, -2.4f - radius);
		makeCircle();
		makekullok();
		makeBody();
	}
	const float F = 18.0f;
	const float m = 1.0f;
	const float g = 9.8f;
	const float rho = 0.8f;
	void moveRight() {
		float derivative = orientation * kb->CalculateDeriative(center.x);
		float alpha = atanf(derivative);
		vec2 V = vec2(cosf(alpha), sinf(alpha));
		float denim = sqrtf(V.x*V.x + V.y * V.y);
		V = V * (1 / (300 * denim));
		float v = (F - m * g*sin(alpha)) / rho;
		moveCenterByVec(V*v);
	}

	void moveLeft() {
		orientation = left;
		float derivative = orientation * kb->CalculateDeriative(center.x);
		float alpha = atanf(derivative);
		vec2 V = vec2(-cosf(alpha), sinf(alpha));
		float denim = sqrtf(V.x*V.x + V.y * V.y);
		V = V * (1 / (300 * denim));
		float v = (F - m * g*sin(alpha)) / rho;
		moveCenterByVec(V*v);
	}

	void fixCentreY() {
		center.y = kb->calculateY(center.x);
		float derivative = orientation * kb->CalculateDeriative(center.x);
		float alpha = atanf(derivative);
		vec2 N = vec2(-sinf(alpha), cosf(alpha));
		float denim = sqrtf(N.x*N.x + N.y * N.y);

		N = N * (1 / denim);
		if (N.y < 0) {
			N = N * (-1);
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
		return mat4(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			wTranslate.x, wTranslate.y, 0, 1); // translation
	}
	mat4 Minv() {
		return mat4(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			-wTranslate.x, -wTranslate.y, 0, 1); // inverse translation
	}
	mat4 Mkullok() {
		return mat4(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1); // translation
	}


	void AddPointByCord(float x, float y) {
		vertexData.push_back(x);
		vertexData.push_back(y);
		vertexData.push_back(0); // red
		vertexData.push_back(0); // green
		vertexData.push_back(0); // blue
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), &vertexData[0], GL_DYNAMIC_DRAW);
		glutPostRedisplay();
	}

	void makeCircle() {
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBindVertexArray(vao);
		for (float i = 0.0f; i <= 2 * PI; i += 2 * PI / 100) {
			AddPointByCord(cos(i)*radius + drawingcentre.x, sin(i)*radius + drawingcentre.y);
		}
	}

	void makekullok() {
		glBindVertexArray(kullovao);
		for (float i = 0.0f; i <= 2 * PI; i += 2 * PI / 6) {
			addPointToBuffer(cos(i + rotate)*radius + drawingcentre.x, sin(i + rotate)*radius + drawingcentre.y, kullodata, vec3(0, 0, 0), kullovbo);
			addPointToBuffer(drawingcentre.x, drawingcentre.y, kullodata, vec3(0, 0, 0), kullovbo);
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
		for (float i = -PI/2; i <= 1.5*PI + 0.001f; i += PI / 14.0f) {
			addPointToBuffer(headcentre.x + 0.4f* cosf(i), headcentre.y + 0.4f*sinf(i), bodydata, vec3(0, 0, 0), vbobody);
		}
	}
	void makeArms() {
		addPointToBuffer(shoulderCenter.x+1.2f, shoulderCenter.y - 2.0f, bodydata, vec3(0, 0, 0), vbobody);
		addPointToBuffer(shoulderCenter.x, shoulderCenter.y, bodydata, vec3(0, 0, 0), vbobody);
		addPointToBuffer(shoulderCenter.x - 1.2f, shoulderCenter.y - 2.0f, bodydata, vec3(0, 0, 0), vbobody);
		addPointToBuffer(shoulderCenter.x, shoulderCenter.y, bodydata, vec3(0, 0, 0), vbobody);
	}

	void makeUpperBody() {
		shoulderCenter.x = drawingcentre.x + 0.9f * (cosf(timePassedSinceStart)) / 3;
		const float c = 2.18f;
		const float b = fabsf(drawingcentre.x - shoulderCenter.x);
		const float a = sqrtf(powf(c, 2) - powf(b, 2));
		shoulderCenter.y = drawingcentre.y + 2.18f + a;// *(PI / 4.0f + (cos(timePassedSinceStart)) / 20);
		addPointToBuffer(shoulderCenter.x, shoulderCenter.y, bodydata, vec3(0, 0, 0), vbobody);
	}

	void makeSeat() {
		addPointToBuffer(drawingcentre.x + 0.3f, drawingcentre.y + 2.18f, bodydata, vec3(0, 0, 0), vbobody);
		addPointToBuffer(drawingcentre.x - 0.3f, drawingcentre.y + 2.18f, bodydata, vec3(0, 0, 0), vbobody);
		addPointToBuffer(drawingcentre.x, drawingcentre.y + 2.18f, bodydata, vec3(0, 0, 0), vbobody);
	}

	const float thighBoneLength = 1.6f;
	const float shinBoneLength = 1.7f;
	
	void makeLeg(const Orientation leg) {
		//based on: https://gist.github.com/jupdike/bfe5eb23d1c395d8a0a1a4ddd94882ac

		const float offset = PI / 2 * leg;
		const vec2 P1 = vec2(drawingcentre.x, drawingcentre.y + 2.2f);
		const vec2 P2 = vec2(cos(rotate + offset)*radius*0.75f + drawingcentre.x, sin(rotate + offset)*radius*0.75f + drawingcentre.y);
		const float R = sqrtf(powf(P2.x - P1.x, 2) + powf(P2.y - P1.y, 2));
		const float centerDx = P1.x - P2.x;
		const float centerDy = P1.y - P2.y;
		const float r = sqrtf(centerDx * centerDx + centerDy * centerDy);
		const float r2d = r * r;
		const float r4d = r2d * r2d;
		const float rThighSquared = thighBoneLength * thighBoneLength;
		const float rShinSquared = shinBoneLength * shinBoneLength;
		const float a = (rThighSquared - rShinSquared) / (2 * r2d);
		const float r2r2 = (rThighSquared - rShinSquared);
		const float c = sqrtf(2 * (rThighSquared + rShinSquared) / r2d - (r2r2 * r2r2) / r4d - 1);
		const float fx = (P1.x + P2.x) / 2 + a * (P2.x - P1.x);
		const float gx = c * (P2.y - P1.y) / 2;
		const float ix1 = fx + gx;
		const float ix2 = fx - gx;
		const float fy = (P1.y + P2.y) / 2 + a * (P2.y - P1.y);
		const float gy = c * (P1.x - P2.x) / 2;
		const float iy1 = fy + gy;
		const float iy2 = fy - gy;

		vec2 asd = orientation == right ? vec2(ix2, iy2) : vec2(ix1, iy1);

		addPointToBuffer(P1.x, P1.y, bodydata, vec3(0, 0, 0), vbobody);
		addPointToBuffer(asd.x, asd.y, bodydata, vec3(0, 0, 0), vbobody);
		addPointToBuffer(P2.x, P2.y, bodydata, vec3(0, 0, 0), vbobody);
		addPointToBuffer(asd.x, asd.y, bodydata, vec3(0, 0, 0), vbobody);
		addPointToBuffer(P1.x, P1.y, bodydata, vec3(0, 0, 0), vbobody);
	}

	void Draw() {
		glLineWidth(3.0f);
		int location = glGetUniformLocation(gpuProgram.getId(), "color");
		clearPreviousData();
		glUniform4f(location, 0.0f, 0.0f, 0.0f, 1.0f);
		makeCircle();

		mat4 MVPTransform = Mkullok() * camera.V() * camera.P();
		MVPTransform.SetUniform(gpuProgram.getId(), "MVP");
		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glDrawArrays(GL_LINE_LOOP, 0, vertexData.size() / 5);


		makekullok();
		MVPTransform = Mkullok() * camera.V() * camera.P();
		MVPTransform.SetUniform(gpuProgram.getId(), "MVP");
		glBindVertexArray(kullovao);
		glBindBuffer(GL_ARRAY_BUFFER, kullovbo);
		glBufferData(GL_ARRAY_BUFFER, kullodata.size() * sizeof(float), &kullodata[0], GL_DYNAMIC_DRAW);
		glDrawArrays(GL_LINE_STRIP, 0, kullodata.size() / 5);


		makeBody();
		MVPTransform = Mkullok() * camera.V() * camera.P();
		MVPTransform.SetUniform(gpuProgram.getId(), "MVP");
		glBindVertexArray(vaobody);
		glBindBuffer(GL_ARRAY_BUFFER, vbobody);
		glDrawArrays(GL_LINE_STRIP, 0, bodydata.size() / 5);
		glLineWidth(2.0f);
	}

	vec2 getCentre() {
		return center;
	}
	float timePassedSinceLastFrame = 0;
	void animate(float et) {
		timePassedSinceStart += et;
		timePassedSinceLastFrame += et;
		if (timePassedSinceLastFrame > 0.02f) {  // 50 FPS max
			timePassedSinceLastFrame = 0;
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
	unsigned short movesSinceLastChange = 0;
	
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
		vertices[0] = vec2(-20.0f, -20.0f); uvs[0] = vec2(0, 0);
		vertices[1] = vec2(20.0f, -20.0f);  uvs[1] = vec2(1, 0);
		vertices[2] = vec2(20.0f, 20.0f);   uvs[2] = vec2(1, 1);
		vertices[3] = vec2(-20.0f, 20.0f);  uvs[3] = vec2(0, 1);
	}
	void Create() {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active
		glGenBuffers(2, vbo);	// Generate 1 vertex buffer objects
		// vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); // make it active, it is an array
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);	   // copy to that part of the memory which will be modified 
		// Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed

		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); // make it active, it is an array
		glBufferData(GL_ARRAY_BUFFER, sizeof(uvs), uvs, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		// Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
	}

	mat4 BikeFollowingMatrice() {
		return mat4(1*appliedZoom,				 0,							0, 0,
					0,							 1*appliedZoom,				0, 0,
					0,					         0,							1, 0,
					biker.getCentre().x / 10,    biker.getCentre().y / 10,  0, 1); // translation
	}
	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source

		mat4 MVPTransform = camera.V() * camera.P();
		if (FollowingBike)
			MVPTransform = MVPTransform * BikeFollowingMatrice();
		// set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
		MVPTransform.SetUniform(backGroundMaker.getId(), "MVP");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

TexturedQuad quad;


void onInitialization() {
	quad.Create();
	kb = new KochanekBartelsSpline(-1.0f, 0.0f, 0.0f);
	biker.Create();
	biker.setSpline(kb);
	glViewport(0, 0, windowWidth, windowHeight);
	glLineWidth(2.0f); // Width of lines in pixels

// Create objects by setting up their vertex data on the GPU

	glGenVertexArrays(1, &vao);	// get 1 vao id
	glBindVertexArray(vao);		// make it active

	unsigned int vbo;		// vertex buffer object
	glGenBuffers(1, &vbo);	// Generate 1 buffer
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	// Geometry with 24 bytes (6 floats or 3 x 2 coordinates)

	//glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);

	glEnableVertexAttribArray(0);  // AttribArray 0
	glVertexAttribPointer(0,       // vbo -> AttribArray 0
		2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
		0, NULL); 		     // stride, offset: tightly packed

	// create program for the GPU  fragmentSourceForBackground
	gpuProgram.Create(vertexSource, fragmentSource, "outColor");
	backGroundMaker.Create(vertexSourceForBackground, fragmentSourceForBackground, "outColor");
	glutPostRedisplay();
}


// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(159 / 256.0f, 207 / 256.0f, 230 / 256.0f, 0.98f);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer

	// Set color to (0, 1, 0) = green
	int location = glGetUniformLocation(gpuProgram.getId(), "color");
	glUniform4f(location, 1.0f, 1.0f, 0.0f, 1.0f); // 3 floats

	float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix, 
							  0, 1, 0, 0,    // row-major!
							  0, 0, 1, 0,
							  0, 0, 0, 1 };

	location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
	glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location

	location = glGetUniformLocation(backGroundMaker.getId(), "MVP");	// Get the GPU location of uniform variable MVP
	glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location

	glBindVertexArray(vao);  // Draw call
	//glDrawArrays(GL_TRIANGLES, 0 /*startIdx*/, 3 /*# Elements*/);
	if (FollowingBike)
		camera.setCentre(biker.getCentre());


	backGroundMaker.Use();
	quad.Draw();
	gpuProgram.Use();
	kb->Draw();
	biker.Draw();
	glutSwapBuffers(); // exchange buffers for double buffering

}
const unsigned char spaceBar = 32;
// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') {
		biker.moveRight();
		glutPostRedisplay();
	}
	if (key == 'w') {
		biker.animate(0.01f);
		glutPostRedisplay();
	}
	if (key == 'a') {
		biker.moveLeft();
		glutPostRedisplay();
	}
	if (key == spaceBar) {
		if (!FollowingBike) {
			camera.Zoom(appliedZoom);
		}
		FollowingBike = !FollowingBike; {
			if (!FollowingBike) {
				camera.setCentre(vec2(0.0f, 0.0f));
				camera.Zoom(1/appliedZoom);

			}
		}

	}
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system


}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
		float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
		float cY = 1.0f - 2.0f * pY / windowHeight;
		kb->AddControlPoint(cX, cY);
		glutPostRedisplay();
	}
}
const float dt = 0.01f;
float lastFrameTime = 0;
bool hadWaited = false;

void onIdle() {
	if (!hadWaited) {
		hadWaited = true;
	}
	static float tend = 0;
	float tstart = tend;
	tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
	int lefutott = 0;
	float elteltido = 0;
	for (float t = tstart; t < tend; t += dt) {
		float Dt = fmin(dt, tend - t);
		biker.animate(Dt);
		lefutott++;
		elteltido += Dt;
	}
	glutPostRedisplay();
}
