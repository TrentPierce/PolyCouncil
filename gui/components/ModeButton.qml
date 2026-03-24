import QtQuick
import QtQuick.Controls

Button {
    property bool selected: false

    implicitHeight: 40
    implicitWidth: 140
    checkable: true
    checked: selected

    contentItem: Text {
        text: parent.text
        color: parent.selected ? Theme.bg0 : Theme.textMain
        font.pixelSize: Qt.application.font.pixelSize ? Qt.application.font.pixelSize * 1.07 : 14
        font.weight: Font.DemiBold
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
    }

    background: Rectangle {
        radius: height / 2
        color: parent.selected ? Theme.mint : Qt.rgba(1, 1, 1, 0.08)
        border.width: 1
        border.color: parent.selected ? Qt.rgba(Theme.mint.r, Theme.mint.g, Theme.mint.b, 0.75) : Qt.rgba(1, 1, 1, 0.1)
        Behavior on color { ColorAnimation { duration: 180; easing.type: Easing.OutCubic } }
    }
}
