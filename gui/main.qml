import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts
import QtQuick.Dialogs
import "components"

ApplicationWindow {
    id: window
    width: 1600
    height: 980
    minimumWidth: 1280
    minimumHeight: 820
    visible: true
    title: "PolyCouncil"

    onClosing: function(close) {
        close.accepted = true
        Qt.quit()
    }

    Material.theme: Material.Dark
    Material.accent: Material.Teal
    Material.primary: Material.BlueGrey
    color: "#0b1020"

    property color bg0: "#0b1020"
    property color bg1: "#121a32"
    property color bg2: "#17203d"
    property color panel: "#1a2340"
    property color panelAlt: "#202c4e"
    property color stroke: "#3b476f"
    property color textMain: "#edf2ff"
    property color textMuted: "#aeb9da"
    property color lavender: "#b7b4ff"
    property color rose: "#ffb7cb"
    property color mint: "#9ce8d8"
    property color peach: "#ffd7b0"
    property color danger: "#ff8f9d"
    property string githubProfileUrl: "https://github.com/TrentPierce"
    property string githubIssuesUrl: "https://github.com/TrentPierce/PolyCouncil/issues"
    readonly property bool compactHeader: width < 1500
    readonly property int shellPadding: width < 1440 ? 16 : 22
    readonly property int sectionSpacing: width < 1440 ? 14 : 18
    readonly property int panelPadding: width < 1440 ? 14 : 18

    function findIndex(options, value) {
        for (let i = 0; i < options.length; ++i) {
            if (options[i].value === value) {
                return i
            }
        }
        return 0
    }

    function formatList(values) {
        if (!values || values.length === 0) {
            return "None"
        }
        return values.join(", ")
    }

    function normalizeSearch(text) {
        return (text || "").toLowerCase().replace(/[^a-z0-9]+/g, " ").replace(/\s+/g, " ").trim()
    }

    Rectangle {
        anchors.fill: parent
        gradient: Gradient {
            GradientStop { position: 0.0; color: bg0 }
            GradientStop { position: 0.55; color: bg1 }
            GradientStop { position: 1.0; color: bg2 }
        }
    }

    Rectangle {
        width: 420
        height: 420
        radius: 210
        x: width - 320
        y: -120
        color: Qt.rgba(183 / 255, 180 / 255, 1, 0.08)
    }

    Rectangle {
        width: 360
        height: 360
        radius: 180
        x: -100
        y: height - 240
        color: Qt.rgba(156 / 255, 232 / 255, 216 / 255, 0.07)
    }

    Drawer {
        id: settingsDrawer
        objectName: "settingsDrawer"
        width: Math.min(520, window.width * 0.4)
        edge: Qt.RightEdge
        modal: false
        interactive: true
        property string selectedPersonaName: ""
        property string pendingPersonaName: ""

        function personaByName(name) {
            const personas = polyBridge.personaLibrary
            for (let i = 0; i < personas.length; ++i) {
                if (personas[i].name === name) {
                    return personas[i]
                }
            }
            return null
        }

        function syncPersonaEditor() {
            const persona = personaByName(selectedPersonaName)
            if (!persona) {
                personaNameField.text = ""
                personaPromptField.text = ""
                return
            }
            personaNameField.text = persona.name
            personaPromptField.text = persona.prompt || ""
        }

        function prepareNewPersona() {
            selectedPersonaName = ""
            pendingPersonaName = ""
            personaNameField.text = ""
            personaPromptField.text = ""
            personaNameField.forceActiveFocus()
        }

        background: GlassPanel {
            color: Qt.rgba(12 / 255, 18 / 255, 33 / 255, 0.94)
            border.color: Qt.rgba(1, 1, 1, 0.08)
        }

        Connections {
            target: polyBridge

            function onPersonasChanged() {
                if (settingsDrawer.pendingPersonaName.length > 0 && settingsDrawer.personaByName(settingsDrawer.pendingPersonaName)) {
                    settingsDrawer.selectedPersonaName = settingsDrawer.pendingPersonaName
                } else if (settingsDrawer.selectedPersonaName.length > 0 && !settingsDrawer.personaByName(settingsDrawer.selectedPersonaName)) {
                    settingsDrawer.selectedPersonaName = ""
                }
                settingsDrawer.pendingPersonaName = ""
                settingsDrawer.syncPersonaEditor()
            }
        }

        ScrollView {
            id: settingsScroll
            anchors.fill: parent
            anchors.margins: 0
            clip: true
            padding: 24
            contentWidth: availableWidth

            ScrollBar.vertical: ScrollBar {
                policy: ScrollBar.AsNeeded
                implicitWidth: 10
                contentItem: Rectangle {
                    implicitWidth: 6
                    radius: width / 2
                    color: Qt.rgba(mint.r, mint.g, mint.b, 0.42)
                }
                background: Rectangle {
                    radius: width / 2
                    color: Qt.rgba(1, 1, 1, 0.06)
                }
            }

            ColumnLayout {
                width: settingsScroll.availableWidth
                spacing: 18

                SectionTitle {
                    eyebrow: "Settings"
                    title: "Run Preferences"
                }

                AppButton {
                    Layout.fillWidth: true
                    tint: rose
                    text: "Report Issue"
                    onClicked: Qt.openUrlExternally(githubIssuesUrl)
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 12
                    Label { text: "Web search"; color: textMain; Layout.fillWidth: true }
                    Switch {
                        checked: polyBridge.webSearchEnabled
                        onToggled: polyBridge.webSearchEnabled = checked
                    }
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 12
                    Label { text: "Personas"; color: textMain; Layout.fillWidth: true }
                    Switch {
                        checked: polyBridge.useRoles
                        onToggled: polyBridge.useRoles = checked
                    }
                }

                ColumnLayout {
                    Layout.fillWidth: true
                    spacing: 8
                    Label { text: "Timeout (seconds)"; color: textMuted }
                    SpinBox {
                        id: timeoutSpin
                        Layout.fillWidth: true
                        from: 15
                        to: 600
                        value: polyBridge.timeoutSeconds
                        onValueModified: polyBridge.timeoutSeconds = value
                    }
                }

                ColumnLayout {
                    Layout.fillWidth: true
                    spacing: 8
                    Label { text: "Max concurrency"; color: textMuted }
                    SpinBox {
                        id: concurrencySpin
                        Layout.fillWidth: true
                        from: 1
                        to: 8
                        value: polyBridge.maxConcurrency
                        onValueModified: polyBridge.maxConcurrency = value
                    }
                }

                GlassPanel {
                    Layout.fillWidth: true
                    implicitHeight: currentRunContent.implicitHeight + 36
                    Layout.preferredHeight: implicitHeight
                    color: Qt.rgba(1, 1, 1, 0.04)

                    ColumnLayout {
                        id: currentRunContent
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.top: parent.top
                        anchors.leftMargin: 18
                        anchors.rightMargin: 18
                        anchors.topMargin: 18
                        spacing: 10

                        Eyebrow { text: "Current Run" }
                        Label {
                            text: polyBridge.runState === "running" ? "Council active" : "Standing by"
                            color: textMain
                            font.pixelSize: Qt.application.font.pixelSize ? Qt.application.font.pixelSize * 1.38 : 18
                            font.weight: Font.DemiBold
                        }
                        Label {
                            Layout.fillWidth: true
                            text: polyBridge.resultsSummary.length > 0 ? polyBridge.resultsSummary : "No finished result yet. The drawer mirrors the current backend state."
                            color: textMuted
                            wrapMode: Text.WordWrap
                        }
                    }
                }

                GlassPanel {
                    Layout.fillWidth: true
                    color: Qt.rgba(1, 1, 1, 0.04)
                    implicitHeight: personaEditorContent.implicitHeight + 36
                    Layout.preferredHeight: implicitHeight

                    ColumnLayout {
                        id: personaEditorContent
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.top: parent.top
                        anchors.leftMargin: 18
                        anchors.rightMargin: 18
                        anchors.topMargin: 18
                        spacing: 12

                        RowLayout {
                            Layout.fillWidth: true

                            SectionTitle {
                                eyebrow: "Personas"
                                title: "Persona Library"
                            }

                            Item { Layout.fillWidth: true }

                            AppButton {
                                tint: lavender
                                text: "New"
                                onClicked: settingsDrawer.prepareNewPersona()
                            }
                        }

                        Label {
                            Layout.fillWidth: true
                            text: "Edit custom personas here, then assign them from the workflow when persona controls are enabled."
                            color: textMuted
                            wrapMode: Text.WordWrap
                        }

                        Column {
                            Layout.fillWidth: true
                            spacing: 8

                            Repeater {
                                model: polyBridge.personaLibrary

                                delegate: Rectangle {
                                    required property var modelData
                                    width: parent.width
                                    height: 50
                                    radius: 18
                                    color: settingsDrawer.selectedPersonaName === modelData.name
                                           ? Qt.rgba(mint.r, mint.g, mint.b, 0.16)
                                           : Qt.rgba(1, 1, 1, 0.04)
                                    border.width: 1
                                    border.color: settingsDrawer.selectedPersonaName === modelData.name
                                                  ? Qt.rgba(mint.r, mint.g, mint.b, 0.42)
                                                  : Qt.rgba(1, 1, 1, 0.08)

                                    RowLayout {
                                        anchors.fill: parent
                                        anchors.margins: 12
            
                                        ColumnLayout {
                                            Layout.fillWidth: true
                                            spacing: 2

                                            Label {
                                                Layout.fillWidth: true
                                                text: modelData.name
                                                color: textMain
                                                font.pixelSize: Qt.application.font.pixelSize ? Qt.application.font.pixelSize * 1.08 : 14
                                                font.weight: Font.DemiBold
                                                elide: Text.ElideRight
                                            }

                                            Label {
                                                Layout.fillWidth: true
                                                text: modelData.builtin
                                                      ? "Built-in persona"
                                                      : modelData.assignmentCount + " model assignments"
                                                color: textMuted
                                                font.pixelSize: Qt.application.font.pixelSize ? Qt.application.font.pixelSize * 0.85 : 11
                                                elide: Text.ElideRight
                                            }
                                        }

                                        SoftBadge {
                                            visible: modelData.builtin
                                            label: "Core"
                                            tint: lavender
                                        }
                                    }

                                    MouseArea {
                                        anchors.fill: parent
                                        onClicked: {
                                            settingsDrawer.selectedPersonaName = modelData.name
                                            settingsDrawer.pendingPersonaName = ""
                                            settingsDrawer.syncPersonaEditor()
                                        }
                                    }
                                }
                            }
                        }

                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 8

                            Label {
                                text: "Persona name"
                                color: textMuted
                            }

                            TextField {
                                id: personaNameField
                                Layout.fillWidth: true
                                enabled: {
                                    const persona = settingsDrawer.personaByName(settingsDrawer.selectedPersonaName)
                                    return !persona || !persona.builtin
                                }
                                placeholderText: "e.g. Systems Optimizer"
                            }
                        }

                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 8

                            Label {
                                text: "System prompt"
                                color: textMuted
                            }

                            TextArea {
                                id: personaPromptField
                                Layout.fillWidth: true
                                Layout.preferredHeight: 160
                                enabled: {
                                    const persona = settingsDrawer.personaByName(settingsDrawer.selectedPersonaName)
                                    return !persona || !persona.builtin
                                }
                                color: textMain
                                placeholderText: "Describe how this persona should reason, challenge assumptions, and write."
                                placeholderTextColor: textMuted
                                wrapMode: TextArea.Wrap
                                leftPadding: 14
                                rightPadding: 14
                                topPadding: 14
                                bottomPadding: 14
                                background: Rectangle {
                                    radius: 18
                                    color: Qt.rgba(1, 1, 1, 0.03)
                                    border.color: personaPromptField.activeFocus ? Qt.rgba(183 / 255, 180 / 255, 1, 0.36) : Qt.rgba(1, 1, 1, 0.08)
                                    border.width: 1
                                }
                            }
                        }

                        RowLayout {
                            Layout.fillWidth: true

                            AppButton {
                                primary: true
                                tint: mint
                                text: settingsDrawer.selectedPersonaName.length > 0 ? "Save Persona" : "Create Persona"
                                enabled: {
                                    const persona = settingsDrawer.personaByName(settingsDrawer.selectedPersonaName)
                                    return personaNameField.text.trim().length > 0 && (!persona || !persona.builtin)
                                }
                                onClicked: {
                                    const nextName = personaNameField.text.trim()
                                    if (settingsDrawer.selectedPersonaName.length > 0) {
                                        const currentPersona = settingsDrawer.personaByName(settingsDrawer.selectedPersonaName)
                                        if (!currentPersona || currentPersona.builtin) {
                                            return
                                        }
                                        settingsDrawer.pendingPersonaName = nextName
                                        polyBridge.update_persona(currentPersona.name, nextName, personaPromptField.text)
                                    } else {
                                        settingsDrawer.pendingPersonaName = nextName
                                        polyBridge.create_persona(nextName, personaPromptField.text)
                                    }
                                }
                            }

                            AppButton {
                                tint: rose
                                text: "Delete"
                                enabled: {
                                    const persona = settingsDrawer.personaByName(settingsDrawer.selectedPersonaName)
                                    return !!persona && !persona.builtin
                                }
                                onClicked: {
                                    const currentPersona = settingsDrawer.personaByName(settingsDrawer.selectedPersonaName)
                                    if (!currentPersona || currentPersona.builtin) {
                                        return
                                    }
                                    settingsDrawer.pendingPersonaName = ""
                                    polyBridge.delete_persona(currentPersona.name)
                                    settingsDrawer.selectedPersonaName = ""
                                    settingsDrawer.syncPersonaEditor()
                                }
                            }
                        }

                        Label {
                            Layout.fillWidth: true
                            text: {
                                const persona = settingsDrawer.personaByName(settingsDrawer.selectedPersonaName)
                                if (!persona) {
                                    return "Create a custom persona here. Built-in personas remain visible in the library for reference."
                                }
                                return persona.builtin
                                       ? "Built-in personas are read-only. Duplicate the prompt into a new custom persona if you want to adapt it."
                                       : "Custom personas can be renamed, updated, and reused across runs."
                            }
                            color: textMuted
                            wrapMode: Text.WordWrap
                        }
                    }
                }
            }
        }
    }

    FileDialog {
        id: attachmentDialog
        title: "Attach Files"
        fileMode: FileDialog.OpenFiles
        onAccepted: {
            for (let i = 0; i < selectedFiles.length; ++i) {
                polyBridge.add_attachment(selectedFiles[i].toString())
            }
        }
    }

    Binding {
        target: providerSelector
        property: "currentIndex"
        value: findIndex(polyBridge.providerOptions, polyBridge.providerType)
        when: providerSelector && !providerSelector.popup.visible
    }

    Binding {
        target: apiServiceSelector
        property: "currentIndex"
        value: findIndex(polyBridge.apiServiceOptions, polyBridge.apiService)
        when: apiServiceSelector && !apiServiceSelector.popup.visible
    }

    Binding {
        target: baseUrlField
        property: "text"
        value: polyBridge.baseUrl
        when: baseUrlField && !baseUrlField.activeFocus
    }

    Binding {
        target: apiKeyField
        property: "text"
        value: polyBridge.apiKey
        when: apiKeyField && !apiKeyField.activeFocus
    }

    Binding {
        target: timeoutSpin
        property: "value"
        value: polyBridge.timeoutSeconds
        when: timeoutSpin && !timeoutSpin.activeFocus
    }

    Binding {
        target: concurrencySpin
        property: "value"
        value: polyBridge.maxConcurrency
        when: concurrencySpin && !concurrencySpin.activeFocus
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: shellPadding
        spacing: sectionSpacing

        GlassPanel {
            Layout.fillWidth: true
            implicitHeight: Math.max(compactHeader ? 132 : 114, headerContent.implicitHeight + panelPadding * 2)
            Layout.preferredHeight: implicitHeight
            Layout.minimumHeight: implicitHeight

            ColumnLayout {
                id: headerContent
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.top: parent.top
                anchors.leftMargin: panelPadding
                anchors.rightMargin: panelPadding
                anchors.topMargin: panelPadding
                spacing: 12

                ColumnLayout {
                    Layout.fillWidth: true
                    spacing: 4

                    Eyebrow { text: "Deliberation Console" }
                    Label {
                        text: "PolyCouncil"
                        color: textMain
                        font.pixelSize: compactHeader ? 28 : 32
                        font.weight: Font.DemiBold
                    }
                    Label {
                        Layout.fillWidth: true
                        text: "Futuristic QML workspace for multi-model voting and collaborative discussions."
                        color: textMuted
                        wrapMode: Text.WordWrap
                    }
                }

                Flow {
                    Layout.fillWidth: true
                    spacing: 12

                    Rectangle {
                        color: Qt.rgba(1, 1, 1, 0.06)
                        radius: 22
                        border.color: Qt.rgba(1, 1, 1, 0.1)
                        border.width: 1
                        height: 44
                        width: compactHeader ? 292 : 300

                        RowLayout {
                            anchors.fill: parent
                            anchors.margins: 3
                            spacing: 4

                            ModeButton {
                                id: deliberationModeButton
                                objectName: "deliberationModeButton"
                                Layout.fillWidth: true
                                text: "Deliberation"
                                selected: polyBridge.mode === "deliberation"
                                onClicked: polyBridge.mode = "deliberation"
                            }

                            ModeButton {
                                id: discussionModeButton
                                objectName: "discussionModeButton"
                                Layout.fillWidth: true
                                text: "Discussion"
                                selected: polyBridge.mode === "discussion"
                                onClicked: polyBridge.mode = "discussion"
                            }
                        }
                    }

                    AppButton {
                        id: settingsButton
                        objectName: "settingsButton"
                        tint: peach
                        text: settingsDrawer.opened ? "Close Settings" : "Settings"
                        onClicked: settingsDrawer.opened ? settingsDrawer.close() : settingsDrawer.open()
                    }

                    SoftBadge { label: polyBridge.selectedCount + " selected"; tint: lavender }
                    SoftBadge { label: polyBridge.totalModels + " models"; tint: mint }
                    SoftBadge { label: polyBridge.runState; tint: rose }
                }
            }
        }

        SplitView {
            id: mainSplit
            Layout.fillWidth: true
            Layout.fillHeight: true
            orientation: Qt.Horizontal

            handle: Rectangle {
                implicitWidth: 8
                color: "transparent"

                Rectangle {
                    anchors.centerIn: parent
                    width: 2
                    height: parent.height - 20
                    radius: 1
                    color: Qt.rgba(1, 1, 1, 0.08)
                }
            }

            GlassPanel {
                SplitView.preferredWidth: Math.max(300, Math.min(380, window.width * 0.22))
                SplitView.minimumWidth: 280
                SplitView.maximumWidth: 430

                ScrollView {
                    id: sidebarScroll
                    anchors.fill: parent
                    anchors.leftMargin: panelPadding
                    anchors.topMargin: panelPadding
                    anchors.bottomMargin: panelPadding
                    anchors.rightMargin: panelPadding + 10
                    clip: true
                    ScrollBar.horizontal.policy: ScrollBar.AlwaysOff

                    ColumnLayout {
                        width: sidebarScroll.availableWidth
                        spacing: sectionSpacing

                        SectionTitle {
                            eyebrow: "Workflow"
                            title: "Connect"
                        }

                        ComboBox {
                            id: providerSelector
                            objectName: "providerSelector"
                            Layout.fillWidth: true
                            model: polyBridge.providerOptions
                            textRole: "label"
                            valueRole: "value"
                            onActivated: polyBridge.providerType = currentValue
                        }

                        ComboBox {
                            id: apiServiceSelector
                            objectName: "apiServiceSelector"
                            Layout.fillWidth: true
                            model: polyBridge.apiServiceOptions
                            textRole: "label"
                            valueRole: "value"
                            onActivated: polyBridge.apiService = currentValue
                        }

                        Label {
                            Layout.fillWidth: true
                            visible: polyBridge.apiService !== "custom"
                            text: "Preset hosted services include OpenAI, OpenRouter, Google Gemini, Anthropic, Groq, Together AI, Kimi, MiniMax, Z.AI, and Fireworks AI."
                            color: textMuted
                            wrapMode: Text.WordWrap
                        }

                        TextField {
                            id: baseUrlField
                            Layout.fillWidth: true
                            color: textMain
                            placeholderText: "Base URL"
                            placeholderTextColor: textMuted
                            leftPadding: 14
                            rightPadding: 14
                            onEditingFinished: polyBridge.baseUrl = text
                        }

                        TextField {
                            id: apiKeyField
                            Layout.fillWidth: true
                            color: textMain
                            echoMode: showKeyCheck.checked ? TextInput.Normal : TextInput.Password
                            placeholderText: "API key"
                            placeholderTextColor: textMuted
                            leftPadding: 14
                            rightPadding: 14
                            onEditingFinished: polyBridge.apiKey = text
                        }

                        CheckBox {
                            id: showKeyCheck
                            text: "Show API key"
                            checked: false
                        }

                        RowLayout {
                            Layout.fillWidth: true

                            AppButton {
                                id: loadModelsButton
                                objectName: "loadModelsButton"
                                Layout.fillWidth: true
                                tint: mint
                                text: polyBridge.isBusy ? "Working..." : "Load Models"
                                enabled: !polyBridge.isBusy
                                onClicked: polyBridge.load_models()
                            }

                            AppButton {
                                id: saveProfileButton
                                objectName: "saveProfileButton"
                                Layout.fillWidth: true
                                tint: lavender
                                text: "Save Profile"
                                onClicked: polyBridge.save_provider_profile()
                            }
                        }

                        RowLayout {
                            Layout.fillWidth: true

                            TextField {
                                id: manualModelField
                                Layout.fillWidth: true
                                color: textMain
                                placeholderText: "Add a model ID manually"
                                placeholderTextColor: textMuted
                                leftPadding: 14
                                rightPadding: 14
                                onAccepted: {
                                    polyBridge.add_manual_model(text)
                                    text = ""
                                }
                            }

                            AppButton {
                                id: addManualModelButton
                                objectName: "addManualModelButton"
                                tint: peach
                                text: "Add Model"
                                onClicked: {
                                    polyBridge.add_manual_model(manualModelField.text)
                                    manualModelField.text = ""
                                }
                            }
                        }

                        Label {
                            Layout.fillWidth: true
                            text: "Use manual add for providers that do not expose a compatible model-list endpoint."
                            color: textMuted
                            wrapMode: Text.WordWrap
                        }

                        GlassPanel {
                            Layout.fillWidth: true
                            color: Qt.rgba(1, 1, 1, 0.035)

                            ColumnLayout {
                                anchors.fill: parent
                                anchors.margins: 16
                                spacing: 12

                                SectionTitle {
                                    eyebrow: "Profiles"
                                    title: "Saved Providers"
                                }

                                Repeater {
                                    model: polyBridge.providerProfiles

                                    delegate: GlassPanel {
                                        required property var modelData
                                        Layout.fillWidth: true
                                        color: Qt.rgba(1, 1, 1, 0.03)

                                        ColumnLayout {
                                            anchors.fill: parent
                                            anchors.margins: 14
                
                                            Label {
                                                Layout.fillWidth: true
                                                text: modelData.summary
                                                color: textMain
                                                wrapMode: Text.WordWrap
                                            }

                                            RowLayout {
                                                Layout.fillWidth: true
                                                spacing: 8
                                                AppButton {
                                                    Layout.fillWidth: true
                                                    tint: mint
                                                    text: "Use"
                                                    onClicked: polyBridge.use_provider_profile(modelData.id)
                                                }
                                                AppButton {
                                                    Layout.fillWidth: true
                                                    tint: rose
                                                    text: "Remove"
                                                    onClicked: polyBridge.remove_provider_profile(modelData.id)
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }




                        SectionTitle {
                            eyebrow: "Workflow"
                            title: "Select Models"
                        }

                        TextField {
                            id: modelSearch
                            Layout.fillWidth: true
                            color: textMain
                            placeholderText: "Filter by model or capability"
                            placeholderTextColor: textMuted
                            leftPadding: 14
                            rightPadding: 14
                        }

                        RowLayout {
                            Layout.fillWidth: true

                            AppButton {
                                id: selectAllButton
                                objectName: "selectAllButton"
                                Layout.fillWidth: true
                                tint: mint
                                text: "Select All"
                                onClicked: polyBridge.select_all()
                            }

                            AppButton {
                                id: clearSelectionButton
                                objectName: "clearSelectionButton"
                                Layout.fillWidth: true
                                tint: rose
                                text: "Clear"
                                onClicked: polyBridge.clear_selection()
                            }
                        }

                        ListView {
                            id: modelList
                            Layout.fillWidth: true
                            Layout.rightMargin: 16
                            Layout.preferredHeight: Math.max(220, Math.min(360, window.height * 0.32))
                            clip: true
                            boundsBehavior: Flickable.StopAtBounds
                            model: polyBridge.models
                            reuseItems: false

                            ScrollBar.vertical: ScrollBar {
                                policy: ScrollBar.AsNeeded
                            }

                            WheelHandler {
                                acceptedDevices: PointerDevice.Mouse | PointerDevice.TouchPad

                                onWheel: function(event) {
                                    const delta = event.pixelDelta.y !== 0 ? event.pixelDelta.y : event.angleDelta.y
                                    const maxY = Math.max(0, modelList.contentHeight - modelList.height)
                                    modelList.contentY = Math.max(0, Math.min(maxY, modelList.contentY - delta))
                                    event.accepted = true
                                }
                            }

                            delegate: Item {
                                required property var modelData
                                width: ListView.view.width
                                property bool isMatch: window.normalizeSearch(modelSearch.text).length === 0 || (modelData.searchText || "").indexOf(window.normalizeSearch(modelSearch.text)) >= 0
                                height: isMatch ? 102 : 0 // 92 + 10 spacing
                                visible: isMatch

                                GlassPanel {
                                    width: parent.width
                                    height: 92
                                    color: modelData.selected ? Qt.rgba(mint.r, mint.g, mint.b, 0.16) : Qt.rgba(1, 1, 1, 0.035)
                                    border.color: modelData.selected ? Qt.rgba(mint.r, mint.g, mint.b, 0.4) : Qt.rgba(1, 1, 1, 0.08)

                                    MouseArea {
                                    anchors.fill: parent
                                    hoverEnabled: true
                                    onClicked: polyBridge.toggle_model(modelData.displayName)
                                    cursorShape: Qt.PointingHandCursor
                                }

                                RowLayout {
                                    anchors.fill: parent
                                    anchors.margins: 14
                                    spacing: 12

                                    Rectangle {
                                        width: 42
                                        height: 42
                                        radius: 21
                                        color: modelData.selected ? Qt.rgba(183 / 255, 180 / 255, 1, 0.3) : Qt.rgba(1, 1, 1, 0.08)
                                        border.color: Qt.rgba(1, 1, 1, 0.16)

                                        Text {
                                            anchors.centerIn: parent
                                            text: modelData.rawModel.charAt(0)
                                            color: textMain
                                            font.pixelSize: Qt.application.font.pixelSize ? Qt.application.font.pixelSize * 1.38 : 18
                                            font.weight: Font.DemiBold
                                        }
                                    }

                                    ColumnLayout {
                                        Layout.fillWidth: true
                                        spacing: 3

                                        Label {
                                            Layout.fillWidth: true
                                            text: modelData.rawModel
                                            color: textMain
                                            elide: Text.ElideRight
                                            font.pixelSize: Qt.application.font.pixelSize ? Qt.application.font.pixelSize * 1.15 : 15
                                            font.weight: Font.DemiBold
                                        }

                                        Label {
                                            Layout.fillWidth: true
                                            text: modelData.capabilitySummary
                                            color: textMuted
                                            elide: Text.ElideRight
                                            font.pixelSize: Qt.application.font.pixelSize ? Qt.application.font.pixelSize * 0.92 : 12
                                        }
                                    }

                                    Rectangle {
                                        width: 20
                                        height: 20
                                        radius: 10
                                        color: modelData.selected ? Qt.rgba(mint.r, mint.g, mint.b, 0.4) : "transparent"
                                        border.color: modelData.selected ? mint : Qt.rgba(1, 1, 1, 0.2)
                                        border.width: 1

                                        Rectangle {
                                            anchors.centerIn: parent
                                            width: 10
                                            height: 10
                                            radius: 5
                                            color: mint
                                            visible: modelData.selected
                                        }
                                    }
                                }
                            }
                            }
                        }

                        SectionTitle {
                            eyebrow: "Workflow"
                            title: "History"
                        }

                        Repeater {
                            model: polyBridge.leaderboard

                            delegate: GlassPanel {
                                required property var modelData
                                Layout.fillWidth: true
                                color: Qt.rgba(1, 1, 1, 0.03)

                                ColumnLayout {
                                    anchors.fill: parent
                                    anchors.margins: 14
                                    spacing: 8

                                    RowLayout {
                                        Layout.fillWidth: true
                                        Label {
                                            Layout.fillWidth: true
                                            text: modelData.label
                                            color: textMain
                                            elide: Text.ElideRight
                                        }
                                        Label {
                                            text: modelData.wins + " wins"
                                            color: textMuted
                                        }
                                    }

                                    ProgressBar {
                                        Layout.fillWidth: true
                                        from: 0
                                        to: 100
                                        value: modelData.share
                                    }
                                }
                            }
                        }

                        SectionTitle {
                            eyebrow: "Replay"
                            title: "Recent Sessions"
                        }

                        Repeater {
                            model: polyBridge.recentSessions

                            delegate: AppButton {
                                required property var modelData
                                Layout.fillWidth: true
                                tint: lavender
                                text: modelData.question.length > 0 ? modelData.question : modelData.name
                                onClicked: polyBridge.replay_session(modelData.path)
                            }
                        }
                    }
                }
            }

            GlassPanel {
                SplitView.fillWidth: true
                SplitView.minimumWidth: 420

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: panelPadding
                    spacing: sectionSpacing

                    GlassPanel {
                        Layout.fillWidth: true
                        implicitHeight: Math.max(compactHeader ? 96 : 84, workspaceStatusRow.implicitHeight + 36)
                        Layout.preferredHeight: implicitHeight
                        Layout.minimumHeight: implicitHeight
                        color: polyBridge.isBusy ? Qt.rgba(156 / 255, 232 / 255, 216 / 255, 0.12) : Qt.rgba(1, 1, 1, 0.03)
                        border.color: polyBridge.isBusy ? Qt.rgba(156 / 255, 232 / 255, 216 / 255, 0.32) : Qt.rgba(1, 1, 1, 0.08)

                        RowLayout {
                            id: workspaceStatusRow
                            anchors.left: parent.left
                            anchors.right: parent.right
                            anchors.top: parent.top
                            anchors.leftMargin: 18
                            anchors.rightMargin: 18
                            anchors.topMargin: 18
                            spacing: 14

                            Rectangle {
                                width: 14
                                height: 14
                                radius: 7
                                color: polyBridge.isBusy ? mint : rose

                                SequentialAnimation on opacity {
                                    running: polyBridge.isBusy
                                    loops: Animation.Infinite
                                    NumberAnimation { from: 1.0; to: 0.3; duration: 520; easing.type: Easing.OutCubic }
                                    NumberAnimation { from: 0.3; to: 1.0; duration: 520; easing.type: Easing.OutCubic }
                                }
                            }

                            ColumnLayout {
                                Layout.fillWidth: true
                                spacing: 4
                                Eyebrow { text: "Workspace" }
                                Label {
                                    text: polyBridge.isBusy ? "Council is running" : "Council is idle"
                                    color: textMain
                                    font.pixelSize: Qt.application.font.pixelSize ? Qt.application.font.pixelSize * 1.38 : 18
                                    font.weight: Font.DemiBold
                                }
                                Label {
                                    Layout.fillWidth: true
                                    text: polyBridge.statusMessage
                                    color: textMuted
                                    wrapMode: Text.WordWrap
                                }
                            }
                        }
                    }

                    SectionTitle {
                        eyebrow: "Council Feed"
                        title: "Live Workspace"
                    }

                    ProgressBar {
                        Layout.fillWidth: true
                        indeterminate: polyBridge.isBusy
                        visible: polyBridge.isBusy
                        value: polyBridge.isBusy ? 0 : 1
                        z: 100
                    }

                    ListView {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        clip: true
                        spacing: 12
                        model: polyBridge.feedEntries

                        delegate: GlassPanel {
                            required property var modelData
                            width: ListView.view.width
                            height: bodyColumn.implicitHeight + 28
                            color: modelData.kind === "user"
                                   ? Qt.rgba(183 / 255, 180 / 255, 1, 0.12)
                                   : modelData.kind === "result"
                                     ? Qt.rgba(156 / 255, 232 / 255, 216 / 255, 0.12)
                                     : Qt.rgba(1, 1, 1, 0.035)

                            ColumnLayout {
                                id: bodyColumn
                                anchors.left: parent.left
                                anchors.right: parent.right
                                anchors.top: parent.top
                                anchors.margins: 14
                                spacing: 6

                                RowLayout {
                                    Layout.fillWidth: true
                                    Label {
                                        Layout.fillWidth: true
                                        text: modelData.title
                                        color: textMain
                                        font.pixelSize: Qt.application.font.pixelSize ? Qt.application.font.pixelSize * 1.15 : 15
                                        font.weight: Font.DemiBold
                                    }
                                    Label {
                                        text: modelData.meta.length > 0 ? modelData.meta + " · " + modelData.timestamp : modelData.timestamp
                                        color: textMuted
                                        font.pixelSize: Qt.application.font.pixelSize ? Qt.application.font.pixelSize * 0.92 : 12
                                    }
                                }

                                Text {
                                    Layout.fillWidth: true
                                    text: modelData.body
                                    color: textMuted
                                    wrapMode: Text.Wrap
                                    textFormat: Text.PlainText
                                }
                            }
                        }

                        ScrollBar.vertical: ScrollBar {}
                    }

                    GlassPanel {
                        Layout.fillWidth: true
                        implicitHeight: composerContent.implicitHeight + 32
                        Layout.preferredHeight: implicitHeight
                        Layout.minimumHeight: implicitHeight
                        color: Qt.rgba(1, 1, 1, 0.045)

                        ColumnLayout {
                            id: composerContent
                            anchors.left: parent.left
                            anchors.right: parent.right
                            anchors.top: parent.top
                            anchors.leftMargin: 16
                            anchors.rightMargin: 16
                            anchors.topMargin: 16
                            spacing: 12

                            RowLayout {
                                Layout.fillWidth: true
                                SectionTitle {
                                    eyebrow: "Composer"
                                    title: "Ask The Council"
                                }
                                Item { Layout.fillWidth: true }
                                SoftBadge { label: polyBridge.selectedModelsText; tint: peach }
                            }

                            ListView {
                                Layout.fillWidth: true
                                Layout.preferredHeight: polyBridge.attachments.length > 0 ? 44 : 0
                                visible: polyBridge.attachments.length > 0
                                orientation: ListView.Horizontal
                                spacing: 8
                                model: polyBridge.attachments

                                delegate: Rectangle {
                                    required property var modelData
                                    height: 34
                                    radius: 17
                                    color: Qt.rgba(183 / 255, 180 / 255, 1, 0.14)
                                    border.color: Qt.rgba(183 / 255, 180 / 255, 1, 0.35)
                                    width: attachmentText.implicitWidth + 56

                                    Text {
                                        id: attachmentText
                                        anchors.verticalCenter: parent.verticalCenter
                                        anchors.left: parent.left
                                        anchors.leftMargin: 14
                                        text: modelData.name
                                        color: textMain
                                        font.pixelSize: Qt.application.font.pixelSize ? Qt.application.font.pixelSize * 0.92 : 12
                                    }

                                    ToolButton {
                                        anchors.verticalCenter: parent.verticalCenter
                                        anchors.right: parent.right
                                        anchors.rightMargin: 6
                                        text: "×"
                                        onClicked: polyBridge.remove_attachment(index)
                                    }
                                }
                            }

                            Item {
                                Layout.fillWidth: true
                                Layout.preferredHeight: Math.max(92, Math.min(160, window.height * 0.15))
                                Layout.minimumHeight: 92

                                Rectangle {
                                    anchors.fill: parent
                                    radius: 18
                                    color: Qt.rgba(1, 1, 1, 0.03)
                                    border.color: promptEditor.activeFocus ? Qt.rgba(156 / 255, 232 / 255, 216 / 255, 0.36) : Qt.rgba(1, 1, 1, 0.08)
                                    border.width: 1
                                }

                                TextArea {
                                    id: promptEditor
                                    anchors.fill: parent
                                    color: textMain
                                    clip: true
                                    topPadding: 30
                                    bottomPadding: 12
                                    leftPadding: 16
                                    rightPadding: 16
                                    wrapMode: TextArea.Wrap
                                    background: Item {}
                                }

                                Text {
                                    visible: promptEditor.length === 0
                                    anchors.left: parent.left
                                    anchors.right: parent.right
                                    anchors.top: parent.top
                                    anchors.leftMargin: 16
                                    anchors.rightMargin: 16
                                    anchors.topMargin: 30
                                    text: "Describe the problem, provide context, and ask the council to deliberate or discuss."
                                    color: textMuted
                                    wrapMode: Text.WordWrap
                                }
                            }

                            RowLayout {
                                Layout.fillWidth: true
    
                                AppButton {
                                    text: "Attach Files"
                                    tint: peach
                                    onClicked: attachmentDialog.open()
                                }

                                AppButton {
                                    text: "Clear Attachments"
                                    tint: rose
                                    enabled: polyBridge.attachments.length > 0
                                    onClicked: polyBridge.clear_attachments()
                                }

                                Item { Layout.fillWidth: true }

                                AppButton {
                                    id: runCouncilButton
                                    objectName: "runCouncilButton"
                                    Layout.preferredWidth: 180
                                    primary: true
                                    tint: mint
                                    text: polyBridge.isBusy ? "Running..." : "Run Council"
                                    enabled: polyBridge.canRun
                                    onClicked: polyBridge.run_council(promptEditor.text)
                                }
                            }
                        }
                    }
                }
            }

            GlassPanel {
                SplitView.preferredWidth: Math.max(340, Math.min(460, window.width * 0.27))
                SplitView.minimumWidth: 320
                SplitView.maximumWidth: 520

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: panelPadding
                    spacing: sectionSpacing

                    RowLayout {
                        Layout.fillWidth: true
                        SectionTitle {
                            eyebrow: "Results"
                            title: "Output Panel"
                        }
                        Item { Layout.fillWidth: true }
                        AppButton {
                            text: "Export JSON"
                            tint: peach
                            onClicked: polyBridge.export_json()
                        }
                    }

                    StackLayout {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        currentIndex: polyBridge.resultsPhase === "pre_run" ? 0 : polyBridge.resultsPhase === "in_run" ? 1 : 2

                        GlassPanel {
                            color: Qt.rgba(1, 1, 1, 0.035)

                            ColumnLayout {
                                anchors.fill: parent
                                anchors.margins: 18
                                spacing: 12

                                Label {
                                    text: "Pre-run configuration"
                                    color: textMain
                                    font.pixelSize: Qt.application.font.pixelSize ? Qt.application.font.pixelSize * 1.54 : 20
                                    font.weight: Font.DemiBold
                                }
                                Label {
                                    Layout.fillWidth: true
                                    text: "Use this panel to confirm the run before you start."
                                    color: textMuted
                                    wrapMode: Text.WordWrap
                                }
                                SoftBadge { label: "Mode: " + polyBridge.runConfig.mode; tint: lavender }
                                SoftBadge { label: "Provider: " + polyBridge.runConfig.providerLabel; tint: mint }
                                SoftBadge { label: "Selected: " + formatList(polyBridge.runConfig.selectedModels); tint: peach }
                                SoftBadge { label: "Attachments: " + formatList(polyBridge.runConfig.attachments); tint: rose }
                                SoftBadge { label: "Timeout: " + polyBridge.runConfig.timeoutSeconds + "s"; tint: lavender }
                                SoftBadge { label: "Concurrency: " + polyBridge.runConfig.maxConcurrency; tint: mint }
                            }
                        }

                        GlassPanel {
                            color: Qt.rgba(156 / 255, 232 / 255, 216 / 255, 0.08)

                            ColumnLayout {
                                anchors.fill: parent
                                anchors.margins: 18
                                spacing: 16

                                BusyIndicator {
                                    Layout.alignment: Qt.AlignHCenter
                                    running: polyBridge.isBusy
                                }

                                Label {
                                    Layout.alignment: Qt.AlignHCenter
                                    text: "Live run state"
                                    color: textMain
                                    font.pixelSize: Qt.application.font.pixelSize ? Qt.application.font.pixelSize * 1.54 : 20
                                    font.weight: Font.DemiBold
                                }

                                Label {
                                    Layout.fillWidth: true
                                    text: polyBridge.statusMessage
                                    color: textMuted
                                    wrapMode: Text.WordWrap
                                }

                                TextArea {
                                    Layout.fillWidth: true
                                    Layout.fillHeight: true
                                    readOnly: true
                                    text: polyBridge.discussionText
                                    color: textMain
                                    wrapMode: TextArea.Wrap
                                    background: Rectangle {
                                        radius: 18
                                        color: Qt.rgba(1, 1, 1, 0.03)
                                        border.color: Qt.rgba(1, 1, 1, 0.08)
                                    }
                                }
                            }
                        }

                        ColumnLayout {
                            spacing: 12

                            GlassPanel {
                                Layout.fillWidth: true
                                implicitHeight: Math.max(140, latestResultContent.implicitHeight + 36)
                                Layout.preferredHeight: implicitHeight
                                Layout.minimumHeight: implicitHeight
                                color: Qt.rgba(1, 1, 1, 0.04)

                                ColumnLayout {
                                    id: latestResultContent
                                    anchors.left: parent.left
                                    anchors.right: parent.right
                                    anchors.top: parent.top
                                    anchors.leftMargin: 18
                                    anchors.rightMargin: 18
                                    anchors.topMargin: 18
                                    spacing: 8

                                    Label {
                                        Layout.fillWidth: true
                                        text: polyBridge.resultsSummary.length > 0 ? polyBridge.resultsSummary : "Latest result"
                                        color: textMain
                                        font.pixelSize: Qt.application.font.pixelSize ? Qt.application.font.pixelSize * 1.54 : 20
                                        font.weight: Font.DemiBold
                                        wrapMode: Text.WordWrap
                                        maximumLineCount: 2
                                        elide: Text.ElideRight
                                    }
                                    Label {
                                        Layout.fillWidth: true
                                        text: polyBridge.lastQuestion.length > 0 ? polyBridge.lastQuestion : "No completed run yet."
                                        color: textMuted
                                        wrapMode: Text.WordWrap
                                    }
                                    SoftBadge {
                                        visible: polyBridge.winnerName.length > 0
                                        label: "Winner: " + polyBridge.winnerName
                                        tint: mint
                                    }
                                }
                            }

                            TabBar {
                                id: resultsTabs
                                Layout.fillWidth: true
                                    background: Rectangle {
                                    color: Qt.rgba(1, 1, 1, 0.035)
                                    radius: 18
                                    border.color: Qt.rgba(1, 1, 1, 0.08)
                                }

                                TabButton {
                                    text: "Winner"
                                    implicitHeight: 42
                                    contentItem: Text {
                                        text: parent.text
                                        color: parent.checked ? bg0 : textMain
                                        font.pixelSize: Qt.application.font.pixelSize ? Qt.application.font.pixelSize * 1.08 : 14
                                        font.weight: Font.DemiBold
                                        horizontalAlignment: Text.AlignHCenter
                                        verticalAlignment: Text.AlignVCenter
                                    }
                                    background: Rectangle {
                                        radius: 16
                                        color: parent.checked ? mint : "transparent"
                                        border.color: parent.checked ? Qt.rgba(mint.r, mint.g, mint.b, 0.8) : "transparent"
                                        border.width: parent.checked ? 1 : 0
                                    }
                                }
                                TabButton {
                                    text: "Answers"
                                    implicitHeight: 42
                                    contentItem: Text {
                                        text: parent.text
                                        color: parent.checked ? bg0 : textMain
                                        font.pixelSize: Qt.application.font.pixelSize ? Qt.application.font.pixelSize * 1.08 : 14
                                        font.weight: Font.DemiBold
                                        horizontalAlignment: Text.AlignHCenter
                                        verticalAlignment: Text.AlignVCenter
                                    }
                                    background: Rectangle {
                                        radius: 16
                                        color: parent.checked ? mint : "transparent"
                                        border.color: parent.checked ? Qt.rgba(mint.r, mint.g, mint.b, 0.8) : "transparent"
                                        border.width: parent.checked ? 1 : 0
                                    }
                                }
                                TabButton {
                                    text: "Ballots"
                                    implicitHeight: 42
                                    contentItem: Text {
                                        text: parent.text
                                        color: parent.checked ? bg0 : textMain
                                        font.pixelSize: Qt.application.font.pixelSize ? Qt.application.font.pixelSize * 1.08 : 14
                                        font.weight: Font.DemiBold
                                        horizontalAlignment: Text.AlignHCenter
                                        verticalAlignment: Text.AlignVCenter
                                    }
                                    background: Rectangle {
                                        radius: 16
                                        color: parent.checked ? mint : "transparent"
                                        border.color: parent.checked ? Qt.rgba(mint.r, mint.g, mint.b, 0.8) : "transparent"
                                        border.width: parent.checked ? 1 : 0
                                    }
                                }

                            }

                            StackLayout {
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                currentIndex: resultsTabs.currentIndex

                                TextArea {
                                    Layout.fillWidth: true
                                    Layout.fillHeight: true
                                    readOnly: true
                                    text: polyBridge.winnerText.length > 0 ? polyBridge.winnerText : "Discussion mode does not produce a single winner."
                                    color: textMain
                                    clip: true
                                    selectByMouse: true
                                    leftPadding: 16
                                    rightPadding: 16
                                    topPadding: 16
                                    bottomPadding: 16
                                    wrapMode: TextArea.Wrap
                                    background: Rectangle {
                                        radius: 18
                                        color: Qt.rgba(1, 1, 1, 0.03)
                                        border.color: Qt.rgba(1, 1, 1, 0.08)
                                    }
                                }

                                TextArea {
                                    Layout.fillWidth: true
                                    Layout.fillHeight: true
                                    readOnly: true
                                    text: polyBridge.answersJson
                                    color: textMain
                                    clip: true
                                    selectByMouse: true
                                    leftPadding: 16
                                    rightPadding: 16
                                    topPadding: 16
                                    bottomPadding: 16
                                    wrapMode: TextArea.Wrap
                                    font.family: "Consolas"
                                    background: Rectangle {
                                        radius: 18
                                        color: Qt.rgba(1, 1, 1, 0.03)
                                        border.color: Qt.rgba(1, 1, 1, 0.08)
                                    }
                                }

                                TextArea {
                                    Layout.fillWidth: true
                                    Layout.fillHeight: true
                                    readOnly: true
                                    text: polyBridge.ballotsJson
                                    color: textMain
                                    clip: true
                                    selectByMouse: true
                                    leftPadding: 16
                                    rightPadding: 16
                                    topPadding: 16
                                    bottomPadding: 16
                                    wrapMode: TextArea.Wrap
                                    font.family: "Consolas"
                                    background: Rectangle {
                                        radius: 18
                                        color: Qt.rgba(1, 1, 1, 0.03)
                                        border.color: Qt.rgba(1, 1, 1, 0.08)
                                    }
                                }


                            }
                        }
                    }
                }
            }
        }

        GlassPanel {
            Layout.fillWidth: true
            Layout.preferredHeight: 54
            color: Qt.rgba(1, 1, 1, 0.045)

            RowLayout {
                anchors.fill: parent
                anchors.margins: 16
                spacing: 12

                Rectangle {
                    width: 10
                    height: 10
                    radius: 5
                    color: polyBridge.isBusy ? mint : lavender
                }

                Label {
                    Layout.fillWidth: true
                    text: polyBridge.statusMessage
                    color: textMuted
                    elide: Text.ElideRight
                }

                SoftBadge { label: "Mode " + polyBridge.mode; tint: lavender }
                SoftBadge { label: polyBridge.selectedCount + "/" + polyBridge.totalModels; tint: mint }

                Label {
                    text: "<a href=\"" + githubProfileUrl + "\">Trent Pierce · GitHub</a>"
                    color: lavender
                    textFormat: Text.RichText
                    linkColor: lavender
                    onLinkActivated: function(link) { Qt.openUrlExternally(link) }
                }
            }
        }
    }
}
