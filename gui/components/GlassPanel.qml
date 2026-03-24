import QtQuick

Rectangle {
    property int accentInset: 18
    property int accentTopMargin: 0
    color: Qt.rgba(1, 1, 1, 0.055)
    radius: 22
    border.width: 1
    border.color: Qt.rgba(1, 1, 1, 0.1)
    layer.enabled: true
    Behavior on color { ColorAnimation { duration: 240; easing.type: Easing.OutCubic } }

    Rectangle {
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.top: parent.top
        anchors.leftMargin: parent.accentInset
        anchors.rightMargin: parent.accentInset
        anchors.topMargin: parent.accentTopMargin
        height: 2
        radius: 2
        gradient: Gradient {
            GradientStop { position: 0.0; color: Qt.rgba(183 / 255, 180 / 255, 1, 0.95) }
            GradientStop { position: 0.5; color: Qt.rgba(156 / 255, 232 / 255, 216 / 255, 0.95) }
            GradientStop { position: 1.0; color: Qt.rgba(1, 183 / 255, 203 / 255, 0.95) }
        }
    }
}
