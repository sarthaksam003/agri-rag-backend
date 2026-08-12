from tokenizers import Tokenizer

from carag.core import get_settings
from carag.services.document_ingestion.chunker import CustomTableAwareChunker

settings = get_settings()

test_str = """<table><tr><td>Control
No.</td><td>Control</td><td>Audit Checks</td><td>Control
Type</td><td>References</td><td>Compliance (Yes/No/NA)</td></tr><tr><td colspan="6">Security Policy</td></tr><tr><td>3.1.1.1</td><td>An information security policy document has been formulated that includes physical, personnel, procedural and technical controls and was submitted to the Controller before commencement of operation.</td><td rowspan="2">1. Obtain the softcopy/hardcopy of the information security policy
2. Check the following
    a. Controller received the policy approved by CA management before commencing operations
    b. Policy contains physical, personnel, procedural and technical controls
    c. information is classified depending</td><td>Mandatory</td><td>IT CA Rules 19.2</td><td>Yes.
IS Policy version 1.0 dated 1st April 2025 is on record. This Policy is preceded by ISP Version 1.0 dated 1st April 2024.</td></tr><tr><td>3.1.1.2</td><td>CA shall ensure the security policy clearly addresses security of both confidential and non-confidential information. The Certifying Authority shall prepare detailed manuals for performing all its activities and shall scrupulously adhere to them.</td><td>Mandatory</td><td>WebTrust 3.9.5, IT Regulations 3</td><td>Yes.
In terms of Information classification Policy of IS Policy version 1.0 dated 1st April 2025, security of both confidential and non-confidential information is detailed.</td></tr><tr><td></td><td></td><td>upon its sensitivity for the CA as per IT Act-2000d. both confidential and non- confidential information addressed in the policye. CA has prepared detailed manuals for performing its activities</td><td></td><td></td><td></td></tr><tr><td>3.1.1.3</td><td>The Certifying Authority shall as approved, in respect of security and risk management controls continuously ensure that security policies and safeguards are in place. Such controls include personnel security and incident handling measures to prevent fraud and security breaches. The security policy shall be approved by management, published and communicated to all employees.</td><td>f. CA ensures security and risk management controls are implementedg. policy has been approved by CA management. Verify by checking approval date.h. policy has been communicated to all employees via mail or training session3. Obtain name of personnel responsible for ensuring that security procedures implementation and verify roles and responsibilities defined</td><td>Mandatory</td><td>IT Regulations 3,WebTrust 3.1.1</td><td>Yes.The CA has approved policies and safeguards are in place.Information Security policy for Media Handling, Disposal and Transit Procedure, Password Management Guidelines, Control of Documents Procedure, Physical and Environmental Security Guidelines, Patch Management,User Access Management, Incident Management, Logging and Auditing, Problem Management, Assets Classification and Labeling,Cryptographic Guidelines, Software Development and</td></tr><tr><td></td><td></td><td rowspan="4">4. Verify the all recent changes done to the policy since last audit5. For sample recent changes done to the policy validate details were submitted to Controller within two weeks</td><td></td><td></td><td>Implementation, Antivirus Management, Asset Management, Backup &amp; Restoration and Change Management is noted.</td></tr><tr><td>3.1.1.4</td><td>An authorized personnel shall be responsible for ensuring that security procedures within their area of responsibility are carried out correctly</td><td>Mandatory</td><td>WebTrust 3.9.8</td><td>Yes.As per TPL (Trusted Personnel List), 21 authorized personnel have been detailed.</td></tr><tr><td>3.1.1.5</td><td>A review process shall be designed for maintaining the security policy.Review dates and responsibilities should be recorded.</td><td>Mandatory</td><td>WebTrust 3.1.3</td><td>Yes.Review process is designed for maintaining the IS Policy version 1.0 dated 1st April 2025.Reviews are undertaken at half yearly intervals.</td></tr><tr><td>3.1.1.6</td><td>Any change made by the Certifying Authority in the security policy shall be submitted by it within two weeks to the Controller</td><td>Mandatory</td><td>IT CA Rules 19</td><td>Yes.No major changes are made in IS Policy version 1.0 dated 1st April 2025 during the period of audit.</td></tr><tr><td colspan="6">Risk Assessment</td></tr><tr><td>3.1.1.7</td><td>The CA's security program shall include an annual Risk Assessment</td><td>1. Obtain copy of annual Risk Assessment plan2. Check the following: a. Threats are identified and</td><td>Mandatory</td><td>IT CA Rules SCHEDULE-Ⅱ 17.3 CA Browser Forum 5</td><td>Yes.An Annual Risk Assessment plan is on record vide Risk Assessment Worksheet dated 15th Feb 2024.</td></tr><tr><td>3.1.1.8</td><td>A list of foreseeable internal and external threats that could result in unauthorized access, disclosure, misuse, alteration, or destruction of any critical information shall be maintained.</td><td rowspan="3">documented as part of risk assessment
b. Standard operating procedures are present for combating threats.
c. controls are in place to prevent misuse, alteration and destruction of sensitive information
d. adequate cyber security tools are implemented by the CA such as DLP, Web Gateway, Email Gateway, Proxy Servers, and Firewalls, SSL/TLS, IDS/IPS, SAST, DAST, etc.
e. a vulnerability assessment by the above-mentioned tools are carried out at a pre decided interval. Report any scheduled VA left out.
f. security plan</td><td>Mandatory</td><td>CA Browser Forum 5</td><td>Yes.
The Risk Assessment worksheet dated 15th Feb 2024 details the list of foreseeable internal and external threats that could result in unauthorized access, disclosure, misuse, alteration, or destruction of critical information.</td></tr><tr><td>3.1.1.9</td><td>The sufficiency of the policies, procedures, information systems, technology, and other arrangements that the CA has in place to counter cyber threats shall be verified.</td><td>Mandatory</td><td>CA Browser Forum 5</td><td>Yes.
The IS Policy version 1.0 dated 1st April 2025 provides for sufficiency of the policies, procedures, information Systems for countering cyber threats. IS Policy version 1.0 dated 1st April 2025 is verified.</td></tr><tr><td>3.1.1.10</td><td>A security plan shall be developed covering the following:
• Security procedures to manage the risks
• administrative, organizational, technical, and physical safeguards
• cost of implementing the specific measures
• security breach plan in case of any incidents</td><td>Mandatory</td><td>CA Browser Forum 5</td><td>Yes.
A security plan has been developed covering Security Procedures to manage administrative, organizational, technical, and physical risks.
The same is documented vide IS Policy version 1.0 dated 1st April 2025.</td></tr><tr><td>3.1.1.11</td><td>Hypervisors, operating system, database, and network device patches and updates shall be applied in a timely manner when deemed necessary based on a risk assessment and follow formal change management procedures</td><td>has been developed
g. updates and patches are applied in timely manner
3. Check the security plan covers the following safeguards:
a. administrative,
b. organizational,
c. technical, and
d. physical
4. Verify formal change management procedures have been developed</td><td>Mandatory</td><td>WebTrust 3.6.18</td><td>Yes.
Formal Change Management procedures are in place for timely patch upduations pertaining to OS, DB and Network devices vide SOP for Change Management ver.2.5 dated 17th April 2024.
Sample CR form dated 6th Sept 2023 is evidenced in Annexure E1.</td></tr><tr><td colspan="6">Self-Assessment</td></tr><tr><td>3.1.1.12</td><td>A documented security self-assessment plan shall be prepared by the CA</td><td rowspan="2">1. Obtain the self-assessment plan developed by the CA
2. Verify on a sample basis the for last two self-assessments conducted
3. Collect report of the self-assessment plan
4. Check the following:
a. Findings of self-assessment were shared with the management</td><td>Mandatory</td><td>IT CA Rules 2.e</td><td>Yes
A documented security self-assessment plan Complied 12.04.2025 and is evidenced.</td></tr><tr><td>3.1.1.13</td><td>Self-Assessment shall be performed on a periodic basis and the findings shall be reported to management and discussed for closure</td><td>Mandatory</td><td>IT CA Rules 2.e</td><td>Yes.
Self-Assessment have been performed and complied. The same is reported to the management.
Internal Audit report for 24.04.2025 is evidenced.</td></tr><tr><td></td><td></td><td>b. Action was taken to address the finding</td><td></td><td></td><td></td></tr><tr><td colspan="6">Training</td></tr><tr><td>3.1.1.14</td><td>The CA shall ensure all personnel performing duties with respect to operation of a CA, ESP, CSP shall receive comprehensive training covering the following:1. CA/ESP/CSP/RA security principles and mechanisms2. All PKI software versions in use on the CA system3. All PKI duties they are expected to perform4. Disaster recovery and business continuity procedures</td><td rowspan="2">1. Obtain the training plan and material for CA employees2. Check the following:a. All relevant topics are covered in the training materialb. Records of training are maintained by CA (e.g. Attendance sheet)c. Refresher training is conductedd. Training is given personnel in case of implementation of every new tool by the CAe. Personnel are trained with the update business continuity policy and procedures at regular intervals</td><td>Mandatory</td><td>X.509 Policy 5.3.3,IT CA Rules SCHEDULE-III.15</td><td>YesIt is ensured that personnel performing duties with respect to operation of a CA, ESP, CSP have received comprehensive training covering CA, PKI, DR &amp; BCP. Training and examsheet evidenced.</td></tr><tr><td>3.1.1.15</td><td>Refresher training must be conducted as and when required, and the Certifying Authority must review these requirements at least once a year. Records of all trainings shall be maintained and reviewed periodically</td><td>Mandatory</td><td>IT CA Rules SCHEDULE III16CA Browser Forum 5.3.3</td><td>Yes.Refrresher trainings are conducted. Training and examsheet evidenced.</td></tr><tr><td>3.1.1.16</td><td>CA shall ensure personnel entrusted with Validation Specialist duties maintain a skill level that enables them to perform such duties satisfactorily</td><td>f. Training requirements are reviewed at least once a year
3. Obtain list of personnel entrusted with validation specialist duties</td><td>Mandatory</td><td>CA Browser Forum 5.3.3</td><td>Yes.
It is ensured that 21 personnel entrusted with Specialist duties have skills that enables them to perform the validation duties satisfactorily.</td></tr><tr><td>3.1.1.17</td><td>All Validation Specialists shall pass an examination provided by the CA on the information verification requirements</td><td>4. Check the following
a. Validation specialist are trained to maintain required skill level
b. All Validation Specialists have passed the examination provided by the CA</td><td>Mandatory</td><td>CA Browser Forum 5.3.3</td><td>Yes.
Validation Specialists have been trained on 11th Feb 2025 and passed the examination on the information verification requirements.</td></tr><tr><td>3.1.1.18</td><td>All individuals responsible for trusted roles shall be made aware of any significant changes in the CA, ESP CSP, or RA operations, as applicable.</td><td>5. Obtain list of personnel responsible for trusted roles
6. Check if the changes made in CA, ESP, CSP Operations were communicated with individuals responsible for trusted roles.
7. Verify only trusted persons are performing verification at all levels</td><td>Mandatory</td><td>X.509 Policy 5.3.4
CA Browser Forum 5.3.4</td><td>Yes.
Individuals responsible for trusted roles are made aware of changes in the CA, ESP CSP and RA operations.</td></tr></table>
"""

tokenizer: Tokenizer = Tokenizer.from_pretrained(settings.EMBEDDINGS_API_MODEL)
chunker = CustomTableAwareChunker(
    tokenizer_or_token_counter=tokenizer,
    chunk_size=200,
    min_characters_per_chunk=100,
)
for i, chunk in enumerate(chunker.chunk(test_str)):
    print(f"CHUNK NUM: {i}")
    print("---------------------------------")
    print(chunk)
    print("---------------------------------")
