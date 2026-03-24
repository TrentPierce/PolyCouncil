import QtQuick
import QtQuick.Controls

Rectangle {
    property string label: ""
    property color tint: Theme.lavender

    radius: 14
    color: Qt.rgba(tint.r, tint.g, tint.b, 0.14)
    border.color: Qt.rgba(tint.r, tint.g, tint.b, 0.32)
    border.width: 1
    implicitHeight: 32
    implicitWidth: badgeText.implicitWidth + 20

    Text {
        id: badgeText
        anchors.centerIn: parent
        text: parent.label
        color: Theme.textMain
        font.pixelSize: Qt.application.font.pixelSize ? Qt.application.font.pixelSize * 0.9 : 12
        font.weight: Font.Medium
    }
}
