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
// Nev    : 
// Neptun : 
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
	
	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(0, 1, 0, 1);	// computed color is the color of the primitive
	}
)";

GPUProgram gpuProgram; // vertex and fragment shaders
unsigned int vao;	   // virtual world on the GPU

class Camera2D {
	vec2 wCenter; // center in world coordinates
	vec2 wSize;   // width and height in world coordinates
public:
	Camera2D() : wCenter(0, 0), wSize(20, 20) { }

	mat4 V() { return TranslateMatrix(-wCenter); }
	mat4 P() { return ScaleMatrix(vec2(2 / wSize.x, 2 / wSize.y)); }

	mat4 Vinv() { return TranslateMatrix(wCenter); }
	mat4 Pinv() { return ScaleMatrix(vec2(wSize.x / 2, wSize.y / 2)); }

	void Zoom(float s) { wSize = wSize * s; }
	void Pan(vec2 t) { wCenter = wCenter + t; }
};

Camera2D camera;		// 2D camera


class LineStrip {
	GLuint				vao, vbo;	// vertex array object, vertex buffer object
	std::vector<float>  vertexData; // interleaved data of coordinates and colors
	vec2			    wTranslate;
public:
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

		vertexData.push_back(-10.0f);
		vertexData.push_back(-3.2f);
		vertexData.push_back(1); // red
		vertexData.push_back(1); // green
		vertexData.push_back(0); // blue
		// copy data to the GPU
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), &vertexData[0], GL_DYNAMIC_DRAW);



		vertexData.push_back(10.0f);
		vertexData.push_back(-3.2f);
		vertexData.push_back(1); // red
		vertexData.push_back(1); // green
		vertexData.push_back(0); // blue
		// copy data to the GPU
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), &vertexData[0], GL_DYNAMIC_DRAW);
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

	void AddPoint(float cX, float cY) {
		// input pipeline
		vec4 wVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv() * Minv();
		// fill interleaved data
		vertexData.push_back(wVertex.x);
		vertexData.push_back(wVertex.y);
		vertexData.push_back(1); // red
		vertexData.push_back(1); // green
		vertexData.push_back(0); // blue
		// copy data to the GPU
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), &vertexData[0], GL_DYNAMIC_DRAW);
	}

	void AddTranslation(vec2 wT) { wTranslate = wTranslate + wT; }

	void Draw() {
		if (vertexData.size() > 0) {
			// set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
			mat4 MVPTransform = M() * camera.V() * camera.P();
			MVPTransform.SetUniform(gpuProgram.getId(), "MVP");
			glBindVertexArray(vao);
			glDrawArrays(GL_LINE_STRIP, 0, vertexData.size() / 5);
			
		}
	}
};
#include <iostream>


int compareVec4ByX(const void* v1, const void* v2){
	int x1 = ((vec4*)v1)->x;
	int x2 = ((vec4*)v2)->x;
	if (x2 < x1)
		return 1;
	if (x2 == x1)
		return 0;
	return -1;
}

class KochanekBartelsSpline {
	unsigned int vaoCurve, vboCurve;
	unsigned int vaoCtrlPoints, vboCtrlPoints;
	int nTesselatedVertices = 100;
	std::vector<float> ts;  // knots
	float tension, bias, continuity;
	float L(int i, float t) {
		float Li = 1.0f;
		for (unsigned int j = 0; j < wCtrlPoints.size(); j++)
			if (j != i) Li *= (t - ts[j]) / (ts[i] - ts[j]);
		return Li;
	}
	float H0(float);
	float H1(float);
	float H2(float);
	float H3(float);
protected:
	std::vector<vec4> wCtrlPoints;		// coordinates of control points
public:
	KochanekBartelsSpline(float t = 0.0f, float c = 0.0f, float b = 0.0f);

	KochanekBartelsSpline(double a, char b) {

	}
	void addSomeControlPoints(float y) {
		AddControlPointByCord(-13.0f, y);
		AddControlPointByCord(-12.0f, y);
		AddControlPointByCord(-10.0f, y);
		AddControlPointByCord(13.0f, y);
		AddControlPointByCord(12.0f, y);
		AddControlPointByCord(10.0f, y);
	 }

	 void AddControlPoint(float cX, float cY) {
		ts.push_back((float)wCtrlPoints.size());
		vec4 wVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
		wCtrlPoints.push_back(wVertex);
		qsort(&wCtrlPoints[0], wCtrlPoints.size(), sizeof(vec4), compareVec4ByX);
		printvector(wCtrlPoints);
	}

	 void AddControlPointByCord(float cX, float cY) {
		 ts.push_back((float)wCtrlPoints.size());
		 vec4 wVertex = vec4(cX, cY, 0, 1);
		 wCtrlPoints.push_back(wVertex);
		 qsort(&wCtrlPoints[0], wCtrlPoints.size(), sizeof(vec4), compareVec4ByX);
		 printvector(wCtrlPoints);
		 std::cout << " Vector ends here" << std::endl;
	 }

	 void printvector(std::vector<vec4> asd) {
		 for (auto temp : asd)
			 std::cout << temp.x << ',' << temp.y << std::endl;
	 }

	void Draw() {
		mat4 VPTransform = camera.V() * camera.P();

		VPTransform.SetUniform(gpuProgram.getId(), "MVP");

		int colorLocation = glGetUniformLocation(gpuProgram.getId(), "color");


		if (wCtrlPoints.size() > 0) {	// draw control points
			glBindVertexArray(vaoCtrlPoints);
			glBindBuffer(GL_ARRAY_BUFFER, vboCtrlPoints);
			glBufferData(GL_ARRAY_BUFFER, wCtrlPoints.size() * 4 * sizeof(float), &wCtrlPoints[0], GL_DYNAMIC_DRAW);
			if (colorLocation >= 0) glUniform3f(colorLocation, 1, 0, 0);
			glPointSize(10.0f);
			glDrawArrays(GL_POINTS, 0, wCtrlPoints.size());
		}

		if (wCtrlPoints.size() >= 2) {	// draw curve
			std::vector<float> vertexData;
		/*	for (int i = 0; i < nTesselatedVertices; i++) {	// Tessellate
				float tNormalized = (float)i / (nTesselatedVertices - 1);
				float t = tStart() + (tEnd() - tStart()) * tNormalized;
				vec4 wVertex = r(t);
				vertexData.push_back(wVertex.x);
				vertexData.push_back(wVertex.y);
			}*/


			for (int x = 0; x < 400; x++) {
				float i = ((float)x / 20.0f) - 10.0f;
				vec4 wVertex(i, calculateY(i), 0, 1);
				vertexData.push_back(wVertex.x);
				vertexData.push_back(wVertex.y);
			}
			// copy data to the GPU
			glBindVertexArray(vaoCurve);
			glBindBuffer(GL_ARRAY_BUFFER, vboCurve);
			glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), &vertexData[0], GL_DYNAMIC_DRAW);
			if (colorLocation >= 0)
				glUniform3f(colorLocation, 1, 1, 0);
			glDrawArrays(GL_LINE_STRIP, 0, 400);

		}

	}

		float calculateY(float x) {
			vec4* leftside = &wCtrlPoints[0];
			vec4* actual = &wCtrlPoints[1];
			vec4* rightside = nullptr;
			vec4* toTheToRight = nullptr;
			for (unsigned int i = 2; i < wCtrlPoints.size() - 1; i++) {
				if (wCtrlPoints[i].x > x ) {
					leftside = &wCtrlPoints[i-2];
					actual = &wCtrlPoints[i-1];
					rightside = &wCtrlPoints[i];
					toTheToRight = &wCtrlPoints[i + 1];
					break;
				}
			}
			std::cout << "x passed: " << x << " actual.x: " << actual->x << std::endl;
		//	std::cout << " points: " << wCtrlPoints.size() << ",  actual:" << actual->x << ", " << actual->y << std::endl;

			/*if (rightside == nullptr)
				return -6.2f;*/
			//Choosing Tangent Vectors
			
			float deltaI = rightside->x - actual->x;
			//incoming target vector
			//float TiI = ((1 - tension)*(1 + continuity)*(1 - bias) / 2)*(rightside->y - actual->y) + ((1 - tension)*(1 - continuity)*(1 + bias) / 2)*(actual->y - leftside->y);
			float TiI = ((1 - tension)*(1 + continuity)*(1 - bias) / 2)*(toTheToRight->y - rightside->y) + ((1 - tension)*(1 - continuity)*(1 + bias) / 2)*(rightside->y - actual->y);

			//outgoing target vec
			float TiO = ((1 - tension)*(1 - continuity)*(1 - bias) / 2)*(rightside->y - actual->y) + ((1 - tension)*(1 + continuity)*(1 + bias) / 2)*(actual->y - leftside->y);

			std::cout << "TiI: " << TiI << ", TiO: " << TiO << std::endl;
			//std::cout << TiI << ", " << TiO << std::endl;
			float res = 0;
			res += H0((x - actual->x) / deltaI)  *  actual->y;
			res += H1((x - actual->x) / deltaI)  *  rightside->y;
			res += H2((x - actual->x) / deltaI)  * 1 * TiO;
			res += H3((x - actual->x) / deltaI)  * 1 * TiI ;
			return res;
			
		}
	float tStart() {
		return ts[0];
	}
	float tEnd() { return ts[wCtrlPoints.size() - 1]; }

	virtual vec4 r(float t) {
		vec4 wPoint = vec4(0, 0, 0, 0);
		for (unsigned int n = 0; n < wCtrlPoints.size(); n++)
			wPoint += wCtrlPoints[n] * L(n, t);
		return wPoint;
	}

};
KochanekBartelsSpline::KochanekBartelsSpline(float t, float c, float b):
	tension(t), continuity(c), bias(b){
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

	addSomeControlPoints(-6.2f);
}
float KochanekBartelsSpline::H0(float s) {
		return 2*s*s*s - 3*s*s + 1;
		}
float KochanekBartelsSpline::H1(float s) {
	return -2 * s*s*s + 3 * s*s ;
}
float KochanekBartelsSpline::H2(float s) {
	return  s*s*s - 2 * s*s + s;
}
float KochanekBartelsSpline::H3(float s) {
	return s*s*s - s*s;
}
LineStrip lineStrip;
KochanekBartelsSpline* kb;
// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glLineWidth(2.0f); // Width of lines in pixels

// Create objects by setting up their vertex data on the GPU
	kb = new KochanekBartelsSpline(-1.0f, 0, 0);
	glGenVertexArrays(1, &vao);	// get 1 vao id
	glBindVertexArray(vao);		// make it active

	unsigned int vbo;		// vertex buffer object
	glGenBuffers(1, &vbo);	// Generate 1 buffer
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	// Geometry with 24 bytes (6 floats or 3 x 2 coordinates)
	

	glEnableVertexAttribArray(0);  // AttribArray 0
	glVertexAttribPointer(0,       // vbo -> AttribArray 0
		2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
		0, NULL); 		     // stride, offset: tightly packed

	// create program for the GPU
	gpuProgram.Create(vertexSource, fragmentSource, "outColor");
	glutPostRedisplay();
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer

	// Set color to (0, 1, 0) = green
	int location = glGetUniformLocation(gpuProgram.getId(), "color");
	glUniform3f(location, 0.0f, 1.0f, 0.0f); // 3 floats

	float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix, 
		                      0, 1, 0, 0,    // row-major!
		                      0, 0, 1, 0,
		                      0, 0, 0, 1 };

	location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
	glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location

	glBindVertexArray(vao);  // Draw call
	//glDrawArrays(GL_TRIANGLES, 0 /*startIdx*/, 3 /*# Elements*/);

	kb->Draw();
	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
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

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}
