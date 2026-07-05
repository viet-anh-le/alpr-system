import assert from 'node:assert/strict'
import { afterEach, describe, it } from 'node:test'

import { lockBodyScroll } from './ui/scrollLock.js'

afterEach(() => {
  delete globalThis.document
})

describe('body scroll lock', () => {
  it('keeps body locked until every nested overlay unlocks', () => {
    globalThis.document = { body: { style: { overflow: 'auto' } } }

    const unlockOuter = lockBodyScroll()
    const unlockInner = lockBodyScroll()

    assert.equal(document.body.style.overflow, 'hidden')
    unlockInner()
    assert.equal(document.body.style.overflow, 'hidden')
    unlockOuter()
    assert.equal(document.body.style.overflow, 'auto')
  })

  it('does not restore stale hidden overflow after nested locks close out of order', () => {
    globalThis.document = { body: { style: { overflow: '' } } }

    const unlockA = lockBodyScroll()
    const unlockB = lockBodyScroll()

    unlockA()
    assert.equal(document.body.style.overflow, 'hidden')
    unlockB()
    assert.equal(document.body.style.overflow, '')
  })
})
