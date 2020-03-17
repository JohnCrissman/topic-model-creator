# topic-model-creator
git-name: chicago-crime-tracker-map

# Topic Model Creator
*Developed by John Crissman

1. Focus
 Predicting social determinants of health
 
2. Data
 On china town patients and produced by patient navigators at Northwestern University

3. Algorithms


# Setup

## Dependencies

This application was developed in Python 3 and HTML, using Visual Studio Code

 - Python 3.6.4
 - Visual Studio Code.  February 2020 (version 1.43)
 - Python packages (including several sub-packages)
	 - `javafx.*`
	 - `org.json.simple.*`
	 - `java.io.*`
	 - `java.net.*`
	 - `java.util.*`
	 - `java.nio.*`
	 - `java.text.*`
	 - `java.time.*`
 - Google Maps API (key embedded within application)


## Installation & Setup

**Install software**
 1. Install [Java 11 from Oracle](https://www.oracle.com/technetwork/java/javase/downloads/jdk11-downloads-5066655.html).  This is a free installation, but you must set up a user account with Oracle.
 2. Install IntelliJ using the [JetBrains Toolbox](https://www.jetbrains.com/toolbox-app/), and be sure to set up Gradle.

**Create a new project**
 1. Open IntelliJ. In the File menu, select New > Project.
 2. Select Gradle in the left menu, and make sure the Project SDK is set to Java 11, and Java is checked. Click Next.
 3. For GroupID, enter `cs.420.neiu`. For ArtifactID, enter `crime.tracker`. Click Next.
 4. Name the project and put it in the desired folder. Click Create.

**Set up the project**
 1. Once the project has loaded, navigate to File > Settings, then Build, Execution, Deployment > Build Tools > Gradle.
	 1. Check "Automatically import this project on changes in build script files."
	 2. "Build and run tests using:" and "Run tests using:" should both be set to Gradle.
	 3. "Use Gradle from" should be set to "'wrapper' task in Gradle build."
	 4. Click OK.
 2. Navigate to File > Project Structure.  Select SDKs on the left side, and make sure Java 11 is selected.  Click OK. 
 3. Navigate to`src` / `main` / `java` in the left side project viewer.
	 1. Right click `java` and select "Open in explorer/finder."
	 2. From the `code.zip` file, copy the `java` and `resources` folders to the project folders (you can write over the empty folders).
 4. Back in IntelliJ, double-click to open `build.gradle`.
	 1. in `plugins`, add two lines:
		  `id 'application'`
                  `id 'org.openjfx.javafxplugin' version '0.0.8'`
       2. Add the following lines:
	       `allprojects { ` 
			`wrapper { gradleVersion = '5.6.2' } }`
		`javafx {`
		`modules = [ 'javafx.controls', 'javafx.web',
		'javafx.graphics'] }`  
		`mainClassName = 'CrimeViewerApplication'`
		`dependencies {`  
			  `testCompile group: 'junit',
			  name: 'junit',
			  version: '4.12'` 
			  `compile group: 'com.googlecode.json-simple',
			  name: 'json-simple',
			  version: '1.1.1' }`
	3. Save the `build.gradle` file and wait for the project build to reload.

**Run the program**
1. On the very far right, click Gradle to expand the Gradle run menu. Expand the menu to Tasks > application > run.  Double-click Run to launch the application.

## Contents

You can rename the current file by clicking the file name in the navigation bar or by clicking the **Rename** button in the file explorer.


## Instructions to Use

The application window contains 3 main parts:

- Top section: search menu
- Middle section: view crimes as a map, list, or set of bar graphs
- Bottom section: change between different views, or exit the application

To use:

- Type in an address (full or partial - this works just like Google Maps to fill in incomplete addresses)
- Select a search radius from the drop-down menu.
- Click "Search" to run your query.

Repeat these steps as many times as desired.

Some crimes are missing latitude and longitude information -- these crimes are excluded from our application, since location information is very important to this application performing as expected!  These crimes are saved in a `logFile_[date information].log` file at application launch, located within `build/resources/main`.
